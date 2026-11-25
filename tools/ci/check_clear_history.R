# Invariant check of clear_history() on every shipped example, in R.
#
# Usage (the package under test installed in the library R uses):
#   Rscript tools/ci/check_clear_history.R [--examples=A,B,...] [--jobs=6]
#       [--timeout=900] [--log-dir=DIR] [--nb=40] [--nk=30] [--seed=11]
#
# One Rscript process per example, at most --jobs at a time; a crash or a run
# past --timeout seconds is a FAIL, and after a crash the example runs again
# without the check that crashed. Constructor and data come from the
# example's `// @example:R` block, evaluated with ai4bayescode_run_chains()
# replaced by a stub that captures model_ctor and ends the block (a block
# without that call: its first new(<Class>, ...), with rng_seed set to
# --seed); keep_history is set to TRUE. With every model built right after
# set.seed(seed) (the BART-family kernels draw from R's RNG),
#     A <- ctor(seed); A$step(nb); A$clear_history(); A$step(nk)
#     B <- ctor(seed); B$step(nb); B$step(nk)   (and a control run B2 like B)
# (a) every get_history() key of A has nk rows (1 when B's has 1) and equals
#     the last rows of B's key bit for bit, dim included; a B2 that differs
#     from B (sampler not reproducible) is reported;
# (b) predict_at(list()): every key of A has nk rows, and a key whose two
#     outputs on B are identical equals the last nk rows of B's bit for bit;
# (c) as (b) at new data: get_dag()$data_inputs set to the constructor
#     arguments of the same name (table NEWDATA otherwise), the first
#     min(10, N) rows when a new N is accepted; keep_tree = TRUE where the
#     constructor has it, with (a) and (b) repeated on those models;
# (d) after ctor(seed); step(nb); clear_history(), get_history() and
#     predict_at(list()) return without an error (row counts recorded).
# A failing (b)/(c) key whose B output lacks nb + nk rows is marked as not one
# row per stored draw. Output: summary.txt, summary.csv and per-example logs
# in --log-dir; exit status 1 when an example fails.

opt <- local({
    a <- commandArgs(trailingOnly = TRUE)
    function(name, default = NULL) {
        hit <- grep(sprintf("^--%s=", name), a, value = TRUE)
        if (length(hit)) sub(sprintf("^--%s=", name), "", hit[length(hit)]) else default
    }
})
`%||%` <- function(x, y) if (is.null(x)) y else x

# predict_at keys that differ from the declared data inputs or from the
# constructor argument names. Each entry maps the constructor arguments
# (v(name) returns the value passed for that C++ parameter) to new data.
NEWDATA <- list(
    # predict_at takes new times "t"; the declared inputs t_new_flag and y are
    # not prediction covariates.
    GPTimeSeries = function(v) list(t = v("t")),
    # Declared inputs y0 = (S0, I0, R0) and t_obs.
    ODE_SIR = function(v) list(y0 = c(v("S0"), v("I0"), v("R0")), t_obs = v("t_obs")),
    # predict_at takes a 0-based g_idx; the constructor takes g_idx_1based.
    HierarchicalLM_joint = function(v) list(X = v("X"), g_idx = v("g_idx_1based") - 1),
    HierarchicalLM_MultivariateRE = function(v) list(X = v("X"), group_idx = v("group")),
    # No declared data inputs; predict_at evaluates S(t) at query times t
    # (and frailty group g, 0-based).
    PehSurvival = function(v) list(t = v("t")),
    PehSharedFrailty = function(v) list(t = v("t"), g = 0)
)
# Examples whose (c) is skipped although data inputs are declared.
NEWDATA_SKIP <- c(
    HSGPRegression = paste("data input 'phi' is the basis matrix the constructor",
                           "computes from x; no constructor argument holds it")
)

# ---- helpers ---------------------------------------------------------------

nrows_of <- function(x) { d <- dim(x); if (is.null(d)) length(x) else d[1L] }
shape_of <- function(x) { d <- dim(x); if (is.null(d)) as.character(length(x)) else paste(d, collapse = "x") }

# The last n entries along the first axis, other axes and dim kept.
last_rows <- function(x, n) {
    len <- nrows_of(x)
    if (n > len) return(NULL)
    idx <- if (n == 0L) integer(0) else seq.int(len - n + 1L, len)
    d <- dim(x)
    if (is.null(d)) return(x[idx])
    do.call(`[`, c(list(x, idx), rep(list(quote(expr = )), length(d) - 1L),
                   list(drop = FALSE)))
}

# The first n entries along the first axis (the new-data subset).
first_rows <- function(x, n) {
    d <- dim(x)
    if (is.null(d)) return(x[seq_len(n)])
    do.call(`[`, c(list(x, seq_len(n)), rep(list(quote(expr = )), length(d) - 1L),
                   list(drop = FALSE)))
}

# Bit-for-bit equality (doubles compared by their bits, attributes included).
same_bits <- function(a, b) !is.null(a) && !is.null(b) && identical(a, b, num.eq = FALSE)

err_msg <- function(e) gsub("\\s+", " ", conditionMessage(e))

# new(Class, <args>) with the argument values bound to symbols, so an error
# message does not deparse the data.
construct <- function(Class, args) {
    env <- new.env(parent = baseenv())
    env$.Class <- Class
    syms <- sprintf(".a%d", seq_along(args))
    for (i in seq_along(args)) assign(syms[i], args[[i]], envir = env)
    argl <- lapply(syms, as.name)
    if (!is.null(names(args))) names(argl) <- names(args)
    eval(as.call(c(list(quote(methods::new), quote(.Class)), argl)), env)
}

arg_names <- function(args) names(args) %||% rep("", length(args))

# The constructor declaration a call matches: named arguments are parameter
# names, positional ones fill the remaining parameters in order, and every
# parameter without a C++ default is supplied.
pick_sig <- function(sigs, args) {
    nms <- arg_names(args); named <- nms[nzchar(nms)]; n_pos <- sum(!nzchar(nms))
    for (s in sigs) {
        if (!all(named %in% s$names)) next
        free <- setdiff(s$names, named)
        if (n_pos > length(free)) next
        covered <- c(named, free[seq_len(n_pos)])
        if (length(setdiff(s$names[!s$has_default], covered))) next
        return(s)
    }
    NULL
}

# Value of parameter p in a call: list(set, value); set = FALSE means the
# C++ default (value) applies.
arg_value <- function(args, sig, p) {
    nms <- arg_names(args)
    if (p %in% nms) return(list(set = TRUE, value = args[[which(nms == p)[1L]]]))
    free <- setdiff(sig$names, nms[nzchar(nms)]); pos <- which(!nzchar(nms))
    k <- match(p, free)
    if (!is.na(k) && k <= length(pos)) return(list(set = TRUE, value = args[[pos[k]]]))
    i <- match(p, sig$names)
    if (!is.na(i) && sig$has_default[i]) return(list(set = FALSE, value = sig$defaults[[i]]))
    list(set = FALSE, value = NULL)
}

set_arg <- function(args, sig, p, value) {
    nms <- arg_names(args)
    if (p %in% nms) { args[[which(nms == p)[1L]]] <- value; return(args) }
    free <- setdiff(sig$names, nms[nzchar(nms)]); pos <- which(!nzchar(nms))
    k <- match(p, free)
    if (!is.na(k) && k <= length(pos)) { args[[pos[k]]] <- value; return(args) }
    args[[p]] <- value     # by name; the named-argument constructor fills the rest
    args
}

# Compare a map of A (history or prediction) with the same map of B.
# B2: a second prediction of B, to tell deterministic keys; NULL for history.
# single: accept 1 row in A when B also has 1 row (single-draw storage).
compare_maps <- function(mA, mB, mB2, nb, nk, single) {
    keys <- sort(union(names(mA), names(mB)))
    out <- list(); bad <- character(0); not_indexed <- character(0)
    for (k in keys) {
        a <- mA[[k]]; b <- mB[[k]]
        if (is.null(a) || is.null(b)) {
            line <- sprintf("%s: returned by %s only", k, if (is.null(a)) "B" else "A")
            out[[k]] <- list(ok = FALSE, line = line); bad <- c(bad, line); next
        }
        ra <- nrows_of(a); rb <- nrows_of(b)
        rows_ok <- ra == nk || (single && ra == 1L && rb == 1L)
        det <- if (is.null(mB2)) TRUE else same_bits(b, mB2[[k]])
        tail_b <- last_rows(b, ra)
        val <- if (!det) NA else if (is.null(tail_b)) FALSE else same_bits(a, tail_b)
        ok <- rows_ok && !isFALSE(val)
        line <- sprintf("%s: A %s, B %s%s%s", k, shape_of(a), shape_of(b),
                        if (!det) ", differs between two B calls (rows checked only)" else "",
                        if (isTRUE(val)) ", values = last rows of B"
                        else if (isFALSE(val)) ", values != last rows of B" else "")
        if (!ok && rb != nb + nk && !(single && rb == 1L)) {
            not_indexed <- c(not_indexed, k)
            line <- sprintf("%s; B rows != nb + nk = %d, so the output is not one row per stored draw",
                            line, nb + nk)
        }
        out[[k]] <- list(rows_A = ra, rows_B = rb, deterministic = det, equal = val,
                         ok = ok, line = line)
        if (!ok) bad <- c(bad, line)
    }
    list(status = if (!length(keys)) "SKIP" else if (length(bad)) "FAIL" else "PASS",
         reason = if (!length(keys)) "no keys returned" else NULL,
         keys = out, failing = bad, not_indexed = not_indexed)
}

hist_safe <- function(m) tryCatch(list(ok = TRUE, value = m$get_history()),
                                  error = function(e) list(ok = FALSE, msg = err_msg(e)))
pred_safe <- function(m, nd) tryCatch(list(ok = TRUE, value = m$predict_at(nd)),
                                      error = function(e) list(ok = FALSE, msg = err_msg(e)))

# B2: a second model built and stepped like B (reproducibility control).
check_history <- function(A, B, B2, nb, nk) {
    hA <- hist_safe(A); hB <- hist_safe(B); hB2 <- hist_safe(B2)
    if (!hA$ok || !hB$ok)
        return(list(status = "FAIL", failing = c(
            if (!hA$ok) paste("A$get_history() error:", hA$msg),
            if (!hB$ok) paste("B$get_history() error:", hB$msg))))
    r <- compare_maps(hA$value, hB$value, NULL, nb, nk, single = TRUE)
    r$reproducible <- hB2$ok && same_bits(hB$value, hB2$value)
    if (!r$reproducible) {
        r$reason <- paste("control: B2, built and stepped like B, gives a different",
                          "history, so the sampler is not reproducible for a fixed seed",
                          "and a value difference does not isolate clear_history()")
        if (r$status == "PASS") r$status <- "FAIL"
    }
    r
}

check_predict <- function(A, B, nd, nb, nk, pB1 = NULL) {
    if (is.null(pB1)) pB1 <- pred_safe(B, nd)
    pB2 <- pred_safe(B, nd); pA <- pred_safe(A, nd)
    if (!pA$ok || !pB1$ok || !pB2$ok)
        return(list(status = "FAIL", failing = c(
            if (!pA$ok) paste("A$predict_at() error:", pA$msg),
            if (!pB1$ok) paste("B$predict_at() error:", pB1$msg),
            if (pB1$ok && !pB2$ok) paste("second B$predict_at() error:", pB2$msg))))
    compare_maps(pA$value, pB1$value, pB2$value, nb, nk, single = FALSE)
}

check_zero_draws <- function(fresh, make, nb) {
    A2 <- fresh(make); A2$step(nb); A2$clear_history()
    h <- hist_safe(A2); p <- pred_safe(A2, list())
    hist_rows <- if (h$ok) vapply(h$value, nrows_of, 1L) else integer(0)
    pred_rows <- if (p$ok) vapply(p$value, nrows_of, 1L) else integer(0)
    bad <- c(if (!h$ok) paste("get_history() error:", h$msg),
             if (!p$ok) paste("predict_at(list()) error:", p$msg))
    by_name <- function(x) if (length(x)) x[order(names(x))] else x
    list(status = if (length(bad)) "FAIL" else "PASS", failing = bad,
         history_rows = by_name(hist_rows), predict_rows = by_name(pred_rows))
}

# ---- one example (child process) -------------------------------------------

capture_ctor <- function(code) {
    E <- new.env(parent = globalenv())
    rec <- new.env(parent = emptyenv()); rec$calls <- list()
    rc_mode <- any(grepl("ai4bayescode_run_chains", code, fixed = TRUE))
    E$ai4bayescode_run_chains <- function(model_ctor, ...)
        stop(structure(class = c("ai4b_captured", "condition"),
                       list(message = "constructor captured", call = NULL, ctor = model_ctor)))
    E$new <- function(Class, ...) {
        args <- list(...)
        rec$calls[[length(rec$calls) + 1L]] <- list(Class = Class, args = args)
        if (!rc_mode)
            stop(structure(class = c("ai4b_captured_new", "condition"),
                           list(message = "new() captured", call = NULL)))
        construct(Class, args)
    }
    got <- tryCatch({
        for (e in parse(text = code, keep.source = FALSE)) eval(e, E)
        list(kind = "none")
    }, ai4b_captured = function(c) list(kind = "run_chains", ctor = c$ctor),
       ai4b_captured_new = function(c) list(kind = "new"),
       error = function(e) list(kind = "error", msg = err_msg(e)))
    got$rec <- rec
    got
}

format_result <- function(res) {
    L <- sprintf("== %s [R]: %s", res$example, res$status %||% "?")
    if (!is.null(res$ctor_args)) L <- c(L, paste0("   constructor: ", res$ctor_args))
    if (length(res$notes)) L <- c(L, paste0("   note: ", res$notes))
    if (!is.null(res$fatal)) L <- c(L, paste0("   ", res$fatal))
    for (s in c("build", "a", "b", "c", "d", "a_kt", "b_kt", "d_kt")) {
        r <- res[[s]]
        if (is.null(r) || (s == "build" && identical(r$status, "PASS"))) next
        lab <- switch(s, a_kt = "a, keep_tree = TRUE", b_kt = "b, keep_tree = TRUE",
                      d_kt = "d, keep_tree = TRUE", s)
        L <- c(L, sprintf("   (%s) %s%s", lab, r$status,
                          if (!is.null(r$reason)) paste0(": ", r$reason) else ""))
        if (!is.null(r$inputs)) L <- c(L, paste0("       inputs: ", r$inputs))
        if (length(r$failing)) L <- c(L, paste0("       ", r$failing))
        if (s %in% c("d", "d_kt") && !is.null(r$history_rows))
            L <- c(L, sprintf("       rows after clear_history(): history {%s}; predict_at {%s}",
                              paste(sprintf("%s=%d", names(r$history_rows), r$history_rows),
                                    collapse = ", "),
                              paste(sprintf("%s=%d", names(r$predict_rows), r$predict_rows),
                                    collapse = ", ")))
    }
    L
}

# The new data of check (c) for the call `args` (constructor signature `sig`):
# list(nd = <named list>, how = <text>) or list(skip = <reason>).
new_data_for <- function(name, A, args, sig) {
    if (name %in% names(NEWDATA_SKIP)) return(list(skip = NEWDATA_SKIP[[name]]))
    v <- function(p) { av <- arg_value(args, sig, p); if (isTRUE(av$set)) av$value else NULL }
    if (name %in% names(NEWDATA)) {
        nd <- NEWDATA[[name]](v); how <- "predict_at keys from the NEWDATA table"
    } else {
        inputs <- as.character(A$get_dag()$data_inputs)
        if (!length(inputs))
            return(list(skip = "no declared data inputs (get_dag()$data_inputs is empty)"))
        nd <- setNames(lapply(inputs, v), inputs); how <- "get_dag()$data_inputs"
    }
    miss <- names(nd)[vapply(nd, function(x) length(x) == 0L, NA)]
    if (length(miss))
        return(list(skip = sprintf("no constructor argument for new-data key(s) %s",
                                   paste(miss, collapse = ", "))))
    nd <- lapply(nd, function(x) { if (is.integer(x) || is.logical(x)) storage.mode(x) <- "double"; x })
    list(nd = nd, how = how)
}

child_main <- function() {
    name <- opt("child"); out <- opt("out")
    nb <- as.integer(opt("nb", "40")); nk <- as.integer(opt("nk", "30"))
    seed <- as.integer(opt("seed", "11"))
    skip <- strsplit(opt("skip", ""), ",", fixed = TRUE)[[1]]
    res <- list(example = name, lang = "R", nb = nb, nk = nk, seed = seed,
                running = NULL, done = character(0), notes = character(0))
    put <- function() saveRDS(res, out)
    # Runs one stage: records it as running (so a crash is attributed to it),
    # stores the returned check result, turns an R error into a FAIL, and
    # replaces a stage that crashed the process in an earlier run by a FAIL.
    stage <- function(s, f) {
        if (s %in% skip) {
            res[[s]] <<- list(status = "FAIL",
                              reason = "crashed the process in an earlier run (see the log)")
        } else {
            res$running <<- s; put(); cat(sprintf("-- stage %s\n", s))
            res[[s]] <<- tryCatch(f(), error = function(e)
                list(status = "FAIL", failing = paste("R error:", err_msg(e))))
        }
        res$done <<- c(res$done, s); res$running <<- NULL; put()
        invisible(res[[s]])
    }
    put()
    options(warn = 1)
    suppressPackageStartupMessages(library(AI4BayesCode))
    cat(sprintf("AI4BayesCode %s from %s\n", packageVersion("AI4BayesCode"),
                find.package("AI4BayesCode")))
    cpp <- ai4bayescode_examples_path(paste0(name, ".cpp"))
    src <- paste(readLines(cpp, warn = FALSE, encoding = "UTF-8"), collapse = "\n")
    code <- AI4BayesCode:::.ai4b_parse_example(src, "R")
    finish <- function() {
        res$status <<- overall_status(res); put()
        cat(format_result(res), sep = "\n")
        invisible()
    }

    # ---- capture: the constructor of the example block
    res$running <- "capture"; put(); cat("-- stage capture\n")
    if (is.null(code)) { res$fatal <- "no // @example:R block"; return(finish()) }
    got <- capture_ctor(code)
    if (got$kind %in% c("none", "error")) {
        res$fatal <- if (got$kind == "error") paste("example block failed:", got$msg)
                     else "example block has neither ai4bayescode_run_chains() nor new()"
        return(finish())
    }
    probe <- NULL
    if (got$kind == "run_chains") {
        probe <- tryCatch(got$ctor(seed), error = function(e) e)
        if (inherits(probe, "error")) {
            res$fatal <- paste("constructor failed:", err_msg(probe)); return(finish())
        }
        res$ctor_source <- "model_ctor of ai4bayescode_run_chains() in the example block"
    } else {
        res$ctor_source <- "first new(<Class>, ...) of the example block (it has no run_chains call)"
    }
    call1 <- if (length(got$rec$calls)) got$rec$calls[[length(got$rec$calls)]] else NULL
    Class <- call1$Class; args <- call1$args
    cls <- if (!is.null(Class)) sub("^Rcpp_", "", Class@.Data) else NA_character_
    sig <- if (!is.null(Class)) pick_sig(AI4BayesCode:::.ai4b_ctor_signatures(cpp, cls), args)
    if (got$kind == "new") {
        if (is.null(sig)) { res$fatal <- "constructor signature not recovered"; return(finish()) }
        old <- arg_value(args, sig, "rng_seed")$value
        args <- set_arg(args, sig, "rng_seed", seed)
        res$notes <- c(res$notes, sprintf("rng_seed %s of the block set to %d",
                                          format(old %||% "default"), seed))
    }
    modified <- got$kind == "new"
    if (!is.null(sig)) {
        kh <- arg_value(args, sig, "keep_history")$value
        if (!isTRUE(as.logical(kh %||% FALSE))) {
            args <- set_arg(args, sig, "keep_history", TRUE); modified <- TRUE
            res$notes <- c(res$notes, "keep_history set to TRUE (the block constructs with FALSE)")
        }
    } else {
        res$notes <- c(res$notes, "constructor arguments not recovered: (c) and keep_tree unavailable")
    }
    make <- if (!modified) function() got$ctor(seed) else function() construct(Class, args)
    res$class <- cls
    if (!is.null(sig))
        res$ctor_args <- sprintf("%s(%s)", cls, paste(vapply(seq_along(args), function(i) {
            nm <- arg_names(args)[i]; x <- args[[i]]
            val <- if (length(x) == 1L && is.atomic(x)) format(x)
                   else paste0("<", class(x)[1L], " ", shape_of(x), ">")
            if (nzchar(nm)) paste0(nm, " = ", val) else val
        }, ""), collapse = ", "))
    cat("constructor:", res$ctor_args %||% "?", "\n")
    res$done <- "capture"; res$running <- NULL; put()

    # ---- build: A (clear_history() after the warmup) and B (no call)
    # Each model is built right after set.seed(seed) and stepped before the
    # next one is built, so models that draw from R's RNG draw the same numbers.
    fresh <- function(mk) { set.seed(seed); mk() }
    run_A <- function(mk) { m <- fresh(mk); m$step(nb); m$clear_history(); m$step(nk); m }
    run_B <- function(mk) { m <- fresh(mk); m$step(nb); m$step(nk); m }
    A <- B <- B2 <- NULL; no_clear <- FALSE
    stage("build", function() {
        m0 <- if (!is.null(probe)) probe else fresh(make)
        if (!tryCatch({ m0$clear_history; TRUE }, error = function(e) FALSE)) {
            no_clear <<- TRUE
            res$fatal <<- "the class has no clear_history() method"
            return(list(status = "FAIL", reason = "the class has no clear_history() method"))
        }
        A <<- run_A(make); B <<- run_B(make); B2 <<- run_B(make)
        list(status = "PASS")
    })
    have <- !is.null(A) && !is.null(B)
    not_run <- function() list(status = "FAIL", reason = "not run: A and B were not built")

    # ---- (a) history
    stage("a", function() if (have) check_history(A, B, B2, nb, nk) else not_run())
    if (isFALSE(res[["a"]]$reproducible))
        res$notes <- c(res$notes, paste("B2 differs from B: value comparisons in (a), (b) and",
                                        "(c) do not isolate clear_history() for this sampler"))

    # ---- (b) prediction at the training data
    stage("b", function() if (have) check_predict(A, B, list(), nb, nk) else not_run())

    # ---- (c) prediction at new data
    make_kt <- NULL
    stage("c", function() {
        if (!have) return(not_run())
        if (is.null(sig)) return(list(status = "SKIP", reason = "constructor arguments not recovered"))
        nd <- new_data_for(name, A, args, sig)
        if (!is.null(nd$skip)) return(list(status = "SKIP", reason = nd$skip))
        how <- nd$how; nd <- nd$nd
        N <- max(vapply(nd, nrows_of, 1L)); n_sub <- min(10L, N)
        sub <- lapply(nd, function(x) if (nrows_of(x) == N) first_rows(x, n_sub) else x)
        flag <- if ("keep_tree" %in% sig$names) "keep_tree" else NULL
        Ac <- A; Bc <- B
        if (!is.null(flag)) {
            args_kt <- set_arg(args, sig, "keep_tree", TRUE)
            make_kt <<- function() construct(Class, args_kt)
            Ac <- run_A(make_kt); Bc <- run_B(make_kt)
            res$a_kt <<- check_history(Ac, Bc, run_B(make_kt), nb, nk)
            res$b_kt <<- check_predict(Ac, Bc, list(), nb, nk)
            put()
        }
        variants <- if (n_sub < N) list(subset = sub, all = nd) else list(all = nd)
        pB1 <- NULL; errs <- character(0); used <- NULL
        for (tn in names(variants)) {
            r <- pred_safe(Bc, variants[[tn]])
            if (r$ok) { pB1 <- r; used <- tn; break }
            errs <- c(errs, sprintf("%s rows: %s", tn, r$msg))
        }
        shown <- if (identical(used, "subset")) sub else nd
        inputs <- sprintf("%s from %s%s%s",
            paste(sprintf("%s %s", names(shown), vapply(shown, shape_of, "")), collapse = ", "),
            how, if (!is.null(used)) sprintf(" (%s)", if (used == "subset")
                sprintf("first %d of N = %d rows", n_sub, N) else sprintf("all N = %d rows", N)) else "",
            if (!is.null(flag)) "; keep_tree = TRUE" else "")
        if (is.null(pB1))
            return(list(status = "SKIP", inputs = inputs, reason = paste(
                "predict_at(new data) fails on B, which never called clear_history():",
                paste(errs, collapse = " | "))))
        c(check_predict(Ac, Bc, variants[[used]], nb, nk, pB1 = pB1), list(inputs = inputs))
    })

    # ---- (d) zero stored draws
    stage("d", function() {
        if (no_clear) return(list(status = "FAIL", reason = "not run: the class has no clear_history() method"))
        r <- check_zero_draws(fresh, make, nb)
        res[["d"]] <<- r; put()
        if (!is.null(make_kt))
            res$d_kt <<- tryCatch(check_zero_draws(fresh, make_kt, nb), error = function(e)
                list(status = "FAIL", failing = paste("R error:", err_msg(e))))
        r
    })
    finish()
}

overall_status <- function(res) {
    if (!is.null(res$fatal)) return("FAIL")
    s <- vapply(c("build", "a", "b", "c", "d", "a_kt", "b_kt", "d_kt"),
                function(k) res[[k]]$status %||% "", "")
    if (any(s == "FAIL")) "FAIL" else "PASS"
}

# ---- driver ----------------------------------------------------------------

this_script <- function() {
    f <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
    normalizePath(f[1L])
}

# The lines of a crashed run's log from R's segfault report on, or its last 8.
crash_excerpt <- function(lines) {
    lines <- trimws(lines[nzchar(trimws(lines))])
    at <- grep("caught segfault|caught bus error|caught illegal", lines)
    if (length(at)) lines[seq.int(at[1L], min(length(lines), at[1L] + 9L))] else tail(lines, 8L)
}

run_example <- function(nm, script, rscript, log_dir, nb, nk, seed, timeout) {
    out <- file.path(log_dir, paste0(nm, ".rds"))
    log <- file.path(log_dir, paste0(nm, ".log"))
    unlink(c(out, log))
    skip <- character(0); crashes <- character(0); t0 <- Sys.time(); res <- NULL
    repeat {
        left <- timeout - as.numeric(difftime(Sys.time(), t0, units = "secs"))
        if (left < 1) { crashes <- c(crashes, "time budget used up by earlier runs"); break }
        unlink(out)
        part <- paste0(log, ".part")
        st <- suppressWarnings(system2(rscript, c("--vanilla", shQuote(script),
            paste0("--child=", nm), paste0("--out=", shQuote(out)),
            sprintf("--nb=%d", nb), sprintf("--nk=%d", nk), sprintf("--seed=%d", seed),
            if (length(skip)) paste0("--skip=", paste(skip, collapse = ","))),
            stdout = part, stderr = part, timeout = ceiling(left)))
        lines <- readLines(part, warn = FALSE); unlink(part)
        cat(c(sprintf("######## run %d (skip: %s), exit status %s", length(skip) + 1L,
                      if (length(skip)) paste(skip, collapse = ",") else "none", st), lines, ""),
            file = log, sep = "\n", append = TRUE)
        res <- if (file.exists(out)) readRDS(out) else NULL
        if (identical(as.integer(st), 0L) && !is.null(res)) break
        stage <- res$running %||% "start"
        what <- if (identical(as.integer(st), 124L)) sprintf("timed out (%d s budget)", timeout)
                else sprintf("process exited with status %s", st)
        crashes <- c(crashes, sprintf("%s during stage '%s'; log: %s", what, stage,
                                      paste(crash_excerpt(lines), collapse = " | ")))
        if (identical(as.integer(st), 124L) || !(stage %in% c("build", "a", "b", "c", "d")) ||
            stage %in% skip) break
        skip <- c(skip, stage)
    }
    if (is.null(res)) res <- list(example = nm, lang = "R")
    if (length(crashes)) {
        res$crashes <- crashes
        res$fatal <- paste(c(res$fatal, crashes), collapse = "; ")
    }
    res$status <- overall_status(res)
    res$seconds <- round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1)
    saveRDS(res, out)
    res
}

driver_main <- function() {
    jobs <- as.integer(opt("jobs", "6")); timeout <- as.numeric(opt("timeout", "900"))
    nb <- as.integer(opt("nb", "40")); nk <- as.integer(opt("nk", "30"))
    seed <- as.integer(opt("seed", "11"))
    log_dir <- opt("log-dir", file.path(tempdir(), "clear-history-R"))
    dir.create(log_dir, recursive = TRUE, showWarnings = FALSE)
    log_dir <- normalizePath(log_dir)
    suppressPackageStartupMessages(library(AI4BayesCode))
    all_ex <- ai4bayescode_list_examples()
    sel <- opt("examples", "")
    ex <- if (nzchar(sel)) strsplit(sel, ",", fixed = TRUE)[[1]] else all_ex
    if (length(bad <- setdiff(ex, all_ex)))
        stop("unknown example(s): ", paste(bad, collapse = ", "))
    script <- this_script()
    rscript <- file.path(R.home("bin"), if (.Platform$OS.type == "windows") "Rscript.exe" else "Rscript")
    cat(sprintf("R %s | AI4BayesCode %s from %s\n%d examples | nb = %d, nk = %d, seed = %d | jobs = %d, timeout = %g s\nlogs: %s\n\n",
                getRversion(), packageVersion("AI4BayesCode"), find.package("AI4BayesCode"),
                length(ex), nb, nk, seed, jobs, timeout, log_dir))
    one <- function(nm) {
        r <- run_example(nm, script, rscript, log_dir, nb, nk, seed, timeout)
        cat(sprintf("%-32s %-5s %7.1fs\n", nm, r$status, r$seconds))
        r
    }
    results <- if (.Platform$OS.type == "windows" || jobs <= 1L) lapply(ex, one)
               else parallel::mclapply(ex, one, mc.cores = jobs, mc.preschedule = FALSE)
    names(results) <- ex
    for (i in seq_along(results))
        if (inherits(results[[i]], "try-error") || is.null(results[[i]]$status))
            results[[i]] <- list(example = ex[i], lang = "R", status = "FAIL",
                                 fatal = paste("driver error:", as.character(results[[i]])))
    st <- function(r, k) r[[k]]$status %||% ""
    tab <- data.frame(
        example = ex,
        status = vapply(results, `[[`, "", "status"),
        a = vapply(results, st, "", "a"), b = vapply(results, st, "", "b"),
        c = vapply(results, st, "", "c"), d = vapply(results, st, "", "d"),
        a_keep_tree = vapply(results, st, "", "a_kt"), b_keep_tree = vapply(results, st, "", "b_kt"),
        d_keep_tree = vapply(results, st, "", "d_kt"),
        c_inputs = vapply(results, function(r) r[["c"]]$inputs %||% "", ""),
        c_reason = vapply(results, function(r) r[["c"]]$reason %||% "", ""),
        seconds = vapply(results, function(r) r$seconds %||% NA_real_, 0),
        fatal = vapply(results, function(r) r$fatal %||% "", ""),
        stringsAsFactors = FALSE)
    write.csv(tab, file.path(log_dir, "summary.csv"), row.names = FALSE)
    txt <- c(sprintf("clear_history() invariant check, R %s, AI4BayesCode %s, nb = %d, nk = %d, seed = %d",
                     getRversion(), packageVersion("AI4BayesCode"), nb, nk, seed), "",
             unlist(lapply(results, function(r) c(format_result(r), ""))))
    writeLines(txt, file.path(log_dir, "summary.txt"))
    n_bad <- sum(tab$status != "PASS")
    cat(sprintf("\n%d of %d examples passed; details in %s\n", nrow(tab) - n_bad, nrow(tab),
                file.path(log_dir, "summary.txt")))
    quit(status = if (n_bad > 0L) 1L else 0L)
}

if (!is.null(opt("child"))) child_main() else driver_main()

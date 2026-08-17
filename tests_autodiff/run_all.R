# ---------------------------------------------------------------------------
# Extended test suite runner.
#
# Almost every file here is an Rcpp::sourceCpp translation unit exporting one
# function (usually named after the file) that returns a list whose entries are
# checks. They were compiled and called BY HAND for a long time, which is how
# test_nuts_adaptation.cpp came to reference an Rcpp::List return type that the
# backend-neutral refactor had already replaced: it stopped compiling and
# nothing noticed for months.
#
#   Rscript tests_autodiff/run_all.R            # everything
#   Rscript tests_autodiff/run_all.R test_ode   # only files matching a pattern
#
# Exit status is 0 only if every file compiled AND every check passed, so this
# is usable as a CI gate. All failures are reported; the run does not stop at
# the first one.
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
    library(Rcpp)
    library(RcppArmadillo)
})

`%||%` <- function(a, b) if (is.null(a)) b else a
.this  <- sub("^--file=", "",
               grep("^--file=", commandArgs(FALSE), value = TRUE))
here   <- normalizePath(if (length(.this)) dirname(.this) else "tests_autodiff")
root    <- normalizePath(file.path(here, ".."))

args    <- commandArgs(trailingOnly = TRUE)
pattern <- if (length(args) > 0) args[1] else NULL

files <- c(list.files(here, pattern = "\\.cpp$", full.names = TRUE),
           list.files(file.path(here, "block_tests"),
                      pattern = "\\.cpp$", full.names = TRUE))
if (!is.null(pattern)) files <- files[grepl(pattern, basename(files))]
files <- sort(files)

# Rcpp's sourceCpp ignores Sys.setenv("PKG_CPPFLAGS"); the supported channel is
# a Makevars file pointed at by R_MAKEVARS_USER (same mechanism
# R/AI4BayesCode_helpers.R uses, and for the same reason).
cppflags <- paste(
    paste0("-I", shQuote(file.path(root, "include"))),
    paste0("-I", shQuote(file.path(root, "include", "mcmclib"))),
    paste0("-I", shQuote(file.path(root, "include", "mcmclib",
                                   "BaseMatrixOps", "include"))),
    paste0("-I", shQuote(file.path(root, "include", "eigen"))),
    paste0("-I", shQuote(file.path(root, "celerite", "include"))),
    paste0("-I", shQuote(file.path(root, "libgp_kernels"))),
    paste0("-I", shQuote(file.path(root, "bart_pure_cpp", "src"))),
    # sourceCpp puts Rcpp's headers on the path automatically but NOT
    # RcppArmadillo's, and the files that include <armadillo> directly (rather
    # than going through block_sampler.hpp's switch) need them.
    paste0("-I", shQuote(system.file("include", package = "RcppArmadillo"))),
    "-DMCMC_ENABLE_ARMA_WRAPPERS", "-DARMA_DONT_USE_WRAPPER")

libs <- if (Sys.info()[["sysname"]] == "Darwin") {
    "-framework Accelerate"
} else {
    "$(BLAS_LIBS) $(LAPACK_LIBS)"
}

mk_file <- tempfile(pattern = "AI4BayesCode_extended_Makevars_")
writeLines(c(paste("PKG_CPPFLAGS =", cppflags),
             paste("PKG_LIBS =", libs)), mk_file)
Sys.setenv(R_MAKEVARS_USER = mk_file)
on.exit(unlink(mk_file), add = TRUE)

# A file "passes" when it compiles and every logical / *_ok / pass entry its
# exported function returns is TRUE. A file that only needs to compile (the
# sanity_* vendored-dependency probes) passes on compiling.
run_one <- function(path) {
    nm  <- tools::file_path_sans_ext(basename(path))
    env <- new.env()
    ok  <- tryCatch({
        sourceCpp(path, env = env, verbose = FALSE, rebuild = FALSE)
        TRUE
    }, error = function(e) {
        cat(sprintf("  COMPILE FAIL: %s\n", conditionMessage(e)))
        FALSE
    })
    if (!isTRUE(ok)) return(list(name = nm, status = "COMPILE-FAIL"))

    fns <- ls(env)
    fns <- fns[vapply(fns, function(f) is.function(get(f, envir = env)), logical(1))]
    # Prefer the same-named entry point; otherwise call every zero-argument
    # exported function (test_wrap_* export check_real_grad / check_simplex_grad).
    entry <- if (nm %in% fns) nm else
             fns[vapply(fns, function(f) length(formals(get(f, envir = env))) == 0,
                        logical(1))]
    if (length(entry) == 0) return(list(name = nm, status = "COMPILE-ONLY"))

    bad <- character(0)
    for (f in entry) {
        res <- tryCatch(get(f, envir = env)(), error = function(e) {
            bad <<- c(bad, sprintf("%s() errored: %s", f, conditionMessage(e)))
            NULL
        })
        if (is.null(res)) next
        flat <- unlist(res, use.names = TRUE)
        # Any logical FALSE, or any entry whose name looks like a verdict.
        for (k in names(flat)) {
            v <- flat[[k]]
            looks_verdict <- grepl("(ok|pass|match|within|valid)$", k,
                                   ignore.case = TRUE)
            if (is.logical(v) && !isTRUE(v)) bad <- c(bad, sprintf("%s$%s is FALSE", f, k))
            else if (looks_verdict && is.numeric(v) && v == 0)
                bad <- c(bad, sprintf("%s$%s is 0", f, k))
        }
    }
    if (length(bad) > 0) {
        for (b in bad) cat(sprintf("  FAIL: %s\n", b))
        return(list(name = nm, status = "FAIL"))
    }
    list(name = nm, status = "PASS")
}

cat(sprintf("=== extended suite: %d files ===\n\n", length(files)))
results <- list()
for (p in files) {
    cat(sprintf("--- %s\n", basename(p)))
    r <- run_one(p)
    results[[length(results) + 1]] <- r
    cat(sprintf("    %s\n\n", r$status))
}

cat("=== SUMMARY ===\n")
for (r in results) cat(sprintf("  %-52s %s\n", r$name, r$status))
n_bad <- sum(vapply(results, function(r) r$status %in% c("FAIL", "COMPILE-FAIL"),
                    logical(1)))
cat(sprintf("\n%d files, %d failing\n", length(results), n_bad))
quit(status = if (n_bad == 0) 0 else 1)

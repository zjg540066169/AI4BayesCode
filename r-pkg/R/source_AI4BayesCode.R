#' Source + load an AI4BayesCode sampler against the installed headers
#'
#' Self-contained, checkout-free counterpart to [Rcpp::sourceCpp()] that
#' sources a generated AI4BayesCode `.cpp` (a **file path** *or* a **source
#' string**) against the header tree shipped inside the installed package
#' (`system.file("include", package = "AI4BayesCode")`). After
#' `install.packages("AI4BayesCode")` and `library(AI4BayesCode)`, this is the
#' only call a user needs: it wires in the full set of include roots, the
#' preprocessor defines, and the explicit platform BLAS/LAPACK libraries that
#' AI4BayesCode samplers require, then loads the declared Rcpp module class
#' into `env`. There is **no dependency on the location of an AI4BayesCode
#' checkout** -- the headers travel with the installed package.
#'
#' Unlike the Rcpp-plugin path -- which
#' could forward only the first `-I` and dropped all `-D`, and therefore
#' sourced only the simplest examples -- this function passes **all** six
#' include roots and **all three** defines as real compiler flags via a
#' temporary `Makevars` pointed at by `R_MAKEVARS_USER`. No source mutation
#' is performed.
#'
#' @section Include roots (resolved from the installed package):
#' \itemize{
#'   \item `include/`                                  (the `AI4BayesCode/...` headers)
#'   \item `include/mcmclib/`
#'   \item `include/mcmclib/BaseMatrixOps/include/`
#'   \item `include/eigen/`                            (for `<Eigen/...>`)
#'   \item `include/celerite/include/`                 (for `<celerite/...>`)
#'   \item `include/libgp_kernels/`                    (for `"libgp_kernels_unity.h"` etc.)
#' }
#'
#' @section Existing user `Makevars` (prepend, not clobber):
#' If you have a personal `~/.R/Makevars` (or `R_MAKEVARS_USER` already set),
#' its settings are preserved: AI4BayesCode's flags are **prepended** to your
#' `PKG_CPPFLAGS` / `PKG_LIBS`, and any other variables you set
#' (`CXX17FLAGS`, OpenMP flags, ...) are passed through verbatim. The override
#' is process-local and restored on exit. (Limitation: a `PKG_CPPFLAGS` /
#' `PKG_LIBS` you define with a line-continuation `\\` or `$(shell ...)` is
#' merged by appending rather than prepending -- correctness is preserved.)
#'
#' @section A file name with a space:
#' Rcpp cannot build a file whose name contains a space, so such a file is
#' compiled from a copy beside it, named `.ai4b_` plus the name with the
#' spaces replaced by underscores (`my model.cpp` gives `.ai4b_my_model.cpp`).
#' The copy is kept, so that an unchanged file is not rebuilt, and can be
#' deleted at any time. In a folder that cannot be written to, the copy goes
#' to R's temporary folder.
#'
#' @param code Either a path to a `.cpp` file, or a length-1 character string
#'   containing the `.cpp` source. A string that is not an existing path is
#'   written to a temporary `.cpp` and sourced.
#' @param rebuild Logical. Force a fresh build, bypassing Rcpp's source
#'   cache. Default `FALSE` (reuse the cache when the source is unchanged).
#' @param verbose Logical. Echo the full compiler command + output. Default
#'   `FALSE`.
#' @param extra_cppflags Character vector of additional flags appended to
#'   `PKG_CPPFLAGS` (e.g. `"-DMY_FLAG"`, extra `-I`).
#' @param extra_libs Character vector of additional flags appended to
#'   `PKG_LIBS`.
#' @param env Environment in which the loaded Rcpp module class(es) are
#'   bound. Defaults to the caller's environment, so
#'   `ai4bayescode_source("Model.cpp"); new(Model, ...)` works at the top level.
#' @param quiet Logical. If `FALSE` (default), print a one-line hint after
#'   loading pointing at [ai4bayescode_doc()] for the class's usage card.
#' @param ... Passed through to [Rcpp::sourceCpp()].
#'
#' @return Invisibly `NULL`. The effect is the side-effect: the Rcpp module
#'   declared in the `.cpp` is loaded into `env`, so `new(<ClassName>, ...)`
#'   becomes available.
#'
#'   [ai4bayescode_example()].
#'
#' @examples
#' \dontrun{
#' library(AI4BayesCode)
#'
#' # From a file -- no AI4BayesCode checkout needed:
#' ai4bayescode_source("./MyModel.cpp")
#' m <- new(MyModel, y, 1L)   # positional: y, rng_seed
#'
#' # From a source string:
#' src <- readLines(ai4bayescode_examples_path("GaussianLocationScale.cpp"))
#' ai4bayescode_source(paste(src, collapse = "\n"))
#' m <- new(GaussianLocationScale, rnorm(100), 1L)   # positional: y, rng_seed
#' }
#' @export
ai4bayescode_source <- function(code,
                                rebuild        = FALSE,
                                verbose        = FALSE,
                                extra_cppflags = character(),
                                extra_libs     = character(),
                                env            = parent.frame(),
                                quiet          = FALSE,
                                ...) {
    if (!requireNamespace("Rcpp", quietly = TRUE)) {
        stop("Rcpp is required but not installed.", call. = FALSE)
    }
    if (!requireNamespace("RcppArmadillo", quietly = TRUE)) {
        stop("RcppArmadillo is required but not installed.", call. = FALSE)
    }

    cpp_file <- .ai4b_resolve_code(code)
    # Rcpp::sourceCpp() cannot build a file whose name contains a space: it
    # refuses one on Windows, and make splits the object name elsewhere.
    # Compile a copy with a space-free name, made beside the original (see
    # .ai4b_stage_spaced_file); the original folder also stays on the include
    # path.
    src_dir <- dirname(cpp_file)
    build_file <- cpp_file
    if (grepl(" ", basename(cpp_file), fixed = TRUE))
        build_file <- .ai4b_stage_spaced_file(cpp_file)

    inc <- system.file("include", package = "AI4BayesCode")
    if (!nzchar(inc) || !dir.exists(inc)) {
        stop("AI4BayesCode include tree not found at system.file('include').\n",
             "The package install looks incomplete -- reinstall AI4BayesCode.",
             call. = FALSE)
    }

    # The six include roots, computed for the *packaged* nesting (celerite and
    # libgp are relocated under include/ by sync_from_core.sh; they are NOT
    # siblings of include/ as in the on-disk checkout).
    roots <- c(
        inc,
        file.path(inc, "mcmclib"),
        file.path(inc, "mcmclib", "BaseMatrixOps", "include"),
        file.path(inc, "eigen"),
        file.path(inc, "celerite", "include"),
        file.path(inc, "libgp_kernels")
    )
    missing <- roots[!dir.exists(roots)]
    if (length(missing)) {
        stop("AI4BayesCode include root(s) missing from the install:\n  ",
             paste(missing, collapse = "\n  "),
             "\nThe package install looks incomplete -- reinstall AI4BayesCode.",
             call. = FALSE)
    }

    # RcppArmadillo bundles <armadillo> + <RcppArmadillo.h>; the examples include
    # <armadillo> and the mcmclib arma wrappers need it, so add its include too
    # (RcppArmadillo is verified present above).
    arma_inc <- system.file("include", package = "RcppArmadillo")
    cppflags <- c(
        paste0("-I", shQuote(c(roots, arma_inc))),
        .ai4b_block_cppflags(),   # installed contributed blocks (ai4bayescode_install_block)
        "-DMCMC_ENABLE_ARMA_WRAPPERS",
        # Trailing "=" makes this an EMPTY replacement, matching
        # RcppArmadillo's own `#define ARMA_DONT_USE_WRAPPER`. A bare -D
        # expands to "1", a different token sequence, and every single user
        # compile then opens with a macro-redefinition warning.
        "-DARMA_DONT_USE_WRAPPER=",
        "-DAI4BAYESCODE_RCPP_MODULE",
        if (!identical(build_file, cpp_file)) paste0("-I", shQuote(src_dir)),
        # Template-heavy samplers can exceed the COFF limit of 2^15 sections.
        if (.Platform$OS.type == "windows") "-Wa,-mbig-obj",
        # exp, log and pow from the Universal C Runtime, several times faster
        # than Rtools' own (see AI4BayesCode/win_ucrt_math.hpp); the matching
        # --wrap is in `libs`.
        if (.Platform$OS.type == "windows")
            "-DAI4B_WIN_UCRT_MATH -include AI4BayesCode/win_ucrt_math.hpp",
        extra_cppflags
    )

    libs <- if (Sys.info()[["sysname"]] == "Darwin") {
        c("-framework Accelerate", extra_libs)
    } else {
        c("$(BLAS_LIBS)", "$(LAPACK_LIBS)",
          if (.Platform$OS.type == "windows") "-Wl,--wrap=exp,--wrap=log,--wrap=pow",
          extra_libs)
    }

    # Rcpp::sourceCpp caches on the SOURCE text; it cannot see that a
    # contributed block's header changed underneath an unchanged -I path. Detect
    # that here and force the rebuild, so an updated block is actually compiled
    # in rather than silently reusing the previous binary.
    fp  <- .ai4b_block_fingerprint()
    key <- normalizePath(cpp_file, winslash = "/", mustWork = FALSE)
    if (nzchar(fp)) {
        prev <- if (exists(key, envir = .ai4b_block_fp_cache, inherits = FALSE))
                    get(key, envir = .ai4b_block_fp_cache) else NULL
        if (!is.null(prev) && !identical(prev, fp)) {
            rebuild <- TRUE
            if (!quiet)
                message("A contributed block changed since the last build of ",
                        basename(cpp_file), " -- recompiling.")
        }
        assign(key, fp, envir = .ai4b_block_fp_cache)
    }

    mk_file <- .ai4b_write_makevars(cppflags, libs)
    on.exit(unlink(mk_file), add = TRUE)

    old_makevars <- Sys.getenv("R_MAKEVARS_USER", unset = NA)
    Sys.setenv(R_MAKEVARS_USER = mk_file)
    on.exit({
        if (is.na(old_makevars)) {
            Sys.unsetenv("R_MAKEVARS_USER")
        } else {
            Sys.setenv(R_MAKEVARS_USER = old_makevars)
        }
    }, add = TRUE)

    # The MinGW-w64 assembler and linker write their temporary files to TMP (or
    # TEMP), not R's TMPDIR, and read paths in the ANSI code page, so a temp
    # folder with other characters (a Chinese user name on an English system)
    # breaks every build. Point both at a spelling of R's temp folder they can
    # open for the build.
    build_tmp <- .ai4b_ansi_tempdir()
    if (!is.null(build_tmp)) {
        old_tmp <- Sys.getenv(c("TMP", "TEMP"), unset = NA)
        Sys.setenv(TMP = build_tmp, TEMP = build_tmp)
        on.exit({
            for (v in names(old_tmp))
                if (is.na(old_tmp[[v]])) Sys.unsetenv(v)
                else do.call(Sys.setenv, stats::setNames(list(old_tmp[[v]]), v))
        }, add = TRUE)
    }
    # On Windows Rcpp normalizes "<file's folder>/../inst/include" whether or not
    # it exists, which warns "The system cannot find the path specified" on
    # every call for a file outside a package source tree. That one is noise.
    withCallingHandlers(
        Rcpp::sourceCpp(build_file, rebuild = rebuild, verbose = verbose, env = env, ...),
        warning = function(w) {
            if (.Platform$OS.type == "windows" &&
                grepl("/inst/include\"", conditionMessage(w), fixed = TRUE))
                invokeRestart("muffleWarning")
        })

    # Register class -> source for ai4bayescode_doc(), and hint the user.
    cls <- tryCatch(.ai4b_doc_register(cpp_file), error = function(e) character(0))
    # Give every exposed class a name-matching constructor, so that
    # new(<Class>, y = y, rng_seed = 7L) means what it reads as and every
    # argument carrying a C++ default may simply be omitted. Best-effort:
    # a class whose signature cannot be recovered keeps stock Rcpp behaviour.
    tryCatch(.ai4b_install_named_ctors(cpp_file, env), error = function(e) character(0))
    if (!quiet && length(cls)) {
        for (nm in cls)
            message(sprintf("\u2713 Loaded '%s' -- run  ai4bayescode_doc(%s)  for usage.",
                            nm, nm))
    }
    invisible(NULL)
}


# Last contributed-block fingerprint seen for each compiled .cpp, per session.
#' @keywords internal
#' @noRd
.ai4b_block_fp_cache <- new.env(parent = emptyenv())

# ---------------------------------------------------------------------------
# Internal: resolve `code` (file path OR source string) to a .cpp path.
# ---------------------------------------------------------------------------
#' @keywords internal
#' @noRd
.ai4b_resolve_code <- function(code) {
    if (!is.character(code) || length(code) != 1L || is.na(code)) {
        stop("`code` must be a single character string: ",
             "a .cpp file path or .cpp source text.", call. = FALSE)
    }

    if (file.exists(code)) {
        return(normalizePath(code, mustWork = TRUE))
    }

    # Not a path -> treat as source text, but guard against a mistyped path by
    # requiring it to look like C++ (newline, #include, or a brace/semicolon).
    looks_like_source <- grepl("\n", code, fixed = TRUE) ||
        grepl("#include", code, fixed = TRUE) ||
        grepl("[{};]", code)
    if (!looks_like_source) {
        stop("`code` is neither an existing file nor recognizable C++ source.\n",
             "If you meant a file path, it does not exist: ", code, call. = FALSE)
    }

    tmp <- tempfile(pattern = "ai4bayes_", fileext = ".cpp")
    writeLines(code, tmp)
    normalizePath(tmp, mustWork = TRUE)
}


# ---------------------------------------------------------------------------
# Internal: build a temp Makevars that PREPENDS our flags onto the user's
# existing Makevars (R_MAKEVARS_USER, else ~/.R/Makevars[.win]) rather than
# replacing it. Returns the temp file path.
# ---------------------------------------------------------------------------
#' @keywords internal
#' @noRd
.ai4b_write_makevars <- function(cppflags, libs) {
    our_cpp <- paste(cppflags, collapse = " ")
    our_lib <- paste(libs,     collapse = " ")

    # Locate the user's effective Makevars, by R's own rule (on Windows
    # Makevars.ucrt, Makevars.win64, Makevars.win, then Makevars).
    user_mk <- tools::makevars_user()
    user_mk <- if (length(user_mk)) user_mk[[1L]] else NA_character_

    out_lines    <- character()
    user_cpp_rhs <- character()
    user_lib_rhs <- character()
    use_fallback <- FALSE

    if (!is.na(user_mk) && file.exists(user_mk)) {
        ul <- readLines(user_mk, warn = FALSE)
        # If the user's PKG_CPPFLAGS/PKG_LIBS use line-continuations or shell
        # expansion, parse-and-prepend is unsafe; fall back to verbatim + `+=`.
        pkg_lines <- grep("^\\s*PKG_(CPPFLAGS|LIBS)\\s*[:+]?=", ul)
        if (any(grepl("\\\\\\s*$", ul[pkg_lines])) ||
            any(grepl("\\$\\(shell", ul[pkg_lines], fixed = FALSE))) {
            use_fallback <- TRUE
        }

        if (use_fallback) {
            warning("AI4BayesCode: your ~/.R/Makevars defines PKG_CPPFLAGS/PKG_LIBS ",
                    "with a line-continuation or $(shell ...); AI4BayesCode flags are ",
                    "appended (not prepended) to preserve correctness.", call. = FALSE)
            out_lines <- c(ul,
                           paste("PKG_CPPFLAGS +=", our_cpp),
                           paste("PKG_LIBS +=",     our_lib))
            mk <- tempfile(pattern = "AI4BayesCode_Makevars_", fileext = "")
            writeLines(out_lines, mk)
            return(mk)
        }

        for (ln in ul) {
            m_cpp <- regmatches(ln, regexec("^\\s*PKG_CPPFLAGS\\s*[:+]?=\\s*(.*)$", ln))[[1]]
            m_lib <- regmatches(ln, regexec("^\\s*PKG_LIBS\\s*[:+]?=\\s*(.*)$", ln))[[1]]
            if (length(m_cpp) == 2L) {
                if (nzchar(trimws(m_cpp[2]))) user_cpp_rhs <- c(user_cpp_rhs, m_cpp[2])
            } else if (length(m_lib) == 2L) {
                if (nzchar(trimws(m_lib[2]))) user_lib_rhs <- c(user_lib_rhs, m_lib[2])
            } else {
                out_lines <- c(out_lines, ln)   # passthrough verbatim
            }
        }
    }

    # Prepend our flags before the user's captured RHS.
    combined_cpp <- paste(c(our_cpp, user_cpp_rhs), collapse = " ")
    combined_lib <- paste(c(our_lib, user_lib_rhs), collapse = " ")

    out_lines <- c(out_lines,
                   paste("PKG_CPPFLAGS =", combined_cpp),
                   paste("PKG_LIBS =",     combined_lib))

    mk <- tempfile(pattern = "AI4BayesCode_Makevars_", fileext = "")
    writeLines(out_lines, mk)
    mk
}



# ---------------------------------------------------------------------------
# Internal: a copy of `cpp_file` under a space-free name, ".ai4b_<name>", in
# the same folder, so that Rcpp's dependency scan finds the file's local
# headers (and compiles their companion .cpp files) as it would for the
# original. It is rewritten only when the text changes, since Rcpp rebuilds
# whenever the file is newer than its last build. In a read-only folder the
# copy goes under tempdir(), keyed by content.
# ---------------------------------------------------------------------------
#' @keywords internal
#' @noRd
.ai4b_stage_spaced_file <- function(cpp_file) {
    name <- gsub("[[:space:]]+", "_", basename(cpp_file))
    dst <- file.path(dirname(cpp_file), paste0(".ai4b_", name))
    same <- file.exists(dst) &&
        identical(unname(tools::md5sum(dst)), unname(tools::md5sum(cpp_file)))
    if (!same && !isTRUE(suppressWarnings(file.copy(cpp_file, dst, overwrite = TRUE)))) {
        key <- unname(tools::md5sum(cpp_file))
        dir <- file.path(tempdir(), "ai4b_src", substr(key, 1L, 16L))
        dir.create(dir, recursive = TRUE, showWarnings = FALSE)
        dst <- file.path(dir, name)
        if (!file.exists(dst)) file.copy(cpp_file, dst)
    }
    normalizePath(dst, winslash = "/", mustWork = TRUE)
}

# ---------------------------------------------------------------------------
# Internal (Windows): TRUE when `path` is plain ASCII. The MinGW-w64 assembler
# and linker read paths in the system's ANSI code page, which plain ASCII
# always fits.
# ---------------------------------------------------------------------------
#' @keywords internal
#' @noRd
.ai4b_ansi_ok <- function(path) !grepl("[^ -~]", path)

# ---------------------------------------------------------------------------
# Internal (Windows): TRUE when the system's ANSI code page can spell `path`,
# so the compiler tools can open it. R's own code page (l10n_info()) is UTF-8
# from R 4.2 on, whatever the system uses, so this reads the system's from
# the registry.
# ---------------------------------------------------------------------------
#' @keywords internal
#' @noRd
.ai4b_system_cp_covers <- function(path) {
    read_reg <- get0("readRegistry", envir = asNamespace("utils"), mode = "function")
    acp <- if (!is.null(read_reg))
        tryCatch(read_reg("SYSTEM\\CurrentControlSet\\Control\\Nls\\CodePage",
                          "HLM")[["ACP"]], error = function(e) NULL)
    !is.null(acp) &&
        (identical(acp, "65001") || !is.na(iconv(enc2utf8(path), "UTF-8", paste0("CP", acp))))
}

# ---------------------------------------------------------------------------
# Internal (Windows): the folder to point TMP and TEMP at for a build: R's
# temp folder, or its 8.3 short name when the long one is not plain ASCII.
# NULL leaves TMP and TEMP as they are (and on other systems).
# ---------------------------------------------------------------------------
#' @keywords internal
#' @noRd
.ai4b_ansi_tempdir <- function() {
    if (.Platform$OS.type != "windows") return(NULL)
    td <- normalizePath(tempdir(), mustWork = FALSE)
    if (.ai4b_ansi_ok(td)) return(td)
    short <- utils::shortPathName(td)
    if (nzchar(short) && .ai4b_ansi_ok(short)) return(short)
    tmp <- Sys.getenv("TMP", Sys.getenv("TEMP"))
    if (!(nzchar(tmp) && (.ai4b_ansi_ok(tmp) || .ai4b_system_cp_covers(tmp))))
        warning("The temporary folder ", td, " has characters outside the Windows ",
                "code page and no short (8.3) name, which the compiler tools need. Set ",
                "TMP and TEMP to a plain-ASCII folder before starting R.", call. = FALSE)
    NULL
}

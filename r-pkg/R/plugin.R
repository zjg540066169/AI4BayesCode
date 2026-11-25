#' Rcpp inline plugin for AI4BayesCode
#'
#' Auto-invoked by [Rcpp::sourceCpp()] when a user `.cpp` contains
#' `// [[Rcpp::depends(AI4BayesCode)]]`. Returns the include paths and
#' preprocessor defines needed to compile against the bundled
#' AI4BayesCode headers + vendored mcmclib / BaseMatrixOps.
#'
#' Users typically do NOT call this directly: use
#' [ai4bayescode_source()], which needs no plugin
#' automatically.
#'
#' The flags include `-include AI4BayesCode/arma_aligned_alloc.hpp`, so, as
#' with [ai4bayescode_source()], Armadillo takes its heap arrays from memory
#' aligned to 64 bytes. Rcpp keys its build cache on the source file alone: a
#' module built through this plugin by an earlier version of AI4BayesCode and
#' kept in a lasting `rcpp.cache.dir` folder is loaded again for an unchanged
#' file unless `rebuild = TRUE` is passed to [Rcpp::sourceCpp()].
#'
#' @return Plugin list per the [Rcpp::Rcpp.plugin.maker()] contract.
#' @export
inlineCxxPlugin <- function() {
    inc          <- system.file("include", package = "AI4BayesCode")
    mcmclib_inc  <- file.path(inc, "mcmclib")
    bmo_inc      <- file.path(inc, "mcmclib", "BaseMatrixOps", "include")
    eigen_inc    <- file.path(inc, "eigen")
    celerite_inc <- file.path(inc, "celerite", "include")
    libgp_inc    <- file.path(inc, "libgp_kernels")

    # Avoid shQuote: Rcpp's PKG_CPPFLAGS parser only picks up the first
    # `-I` when paths are single-quoted. Plain unquoted -I works for paths
    # without spaces. On Windows the default user library sits under
    # C:/Users/<account>/, which may contain one, so use Rcpp's own convention
    # there: the 8.3 short name, with forward slashes. Mirror the six include
    # roots resolved by ai4bayescode_source() so examples needing eigen /
    # celerite / libgp also compile via the Rcpp plugin path.
    build_path <- function(p) {
        if (.Platform$OS.type == "windows" && grepl(" ", p, fixed = TRUE))
            p <- utils::shortPathName(p)
        chartr("\\", "/", p)
    }
    cppflags <- paste(
        paste0("-I", build_path(inc)),
        paste0("-I", build_path(mcmclib_inc)),
        paste0("-I", build_path(bmo_inc)),
        paste0("-I", build_path(eigen_inc)),
        paste0("-I", build_path(celerite_inc)),
        paste0("-I", build_path(libgp_inc)),
        "-DMCMC_ENABLE_ARMA_WRAPPERS",
        # Trailing "=" makes this an EMPTY replacement, matching
        # RcppArmadillo's own `#define ARMA_DONT_USE_WRAPPER`. A bare -D
        # expands to "1", a different token sequence, and every single user
        # compile then opens with a macro-redefinition warning.
        "-DARMA_DONT_USE_WRAPPER=",
        "-DAI4BAYESCODE_RCPP_MODULE",
        # Armadillo takes its heap arrays from 64-byte-aligned memory, as in
        # ai4bayescode_source() (see AI4BayesCode/arma_aligned_alloc.hpp).
        # -include reads the header before the first line of the source, so
        # before <RcppArmadillo.h>. The relative name needs no quoting: the
        # compiler looks for it in its working directory (Rcpp's build
        # folder), then in R's include folder, then under the first -I above,
        # and only the last has an AI4BayesCode folder.
        "-include AI4BayesCode/arma_aligned_alloc.hpp"
    )

    list(
        includes = "",
        env = list(PKG_CPPFLAGS = cppflags)
    )
}

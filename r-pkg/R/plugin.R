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
        "-DAI4BAYESCODE_RCPP_MODULE"
    )

    list(
        includes = "",
        env = list(PKG_CPPFLAGS = cppflags)
    )
}

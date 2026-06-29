#' Rcpp inline plugin for AI4BayesCode
#'
#' Auto-invoked by [Rcpp::sourceCpp()] when a user `.cpp` contains
#' `// [[Rcpp::depends(AI4BayesCode)]]`. Returns the include paths and
#' preprocessor defines needed to compile against the bundled
#' AI4BayesCode headers + vendored mcmclib / BaseMatrixOps.
#'
#' Users typically do NOT call this directly: use
#' [ai4bayescode_sourceCpp()] which injects the depends directive
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
    # without spaces (R library paths typically have no spaces). Mirror the
    # six include roots resolved by source_AI4BayesCode() so examples needing
    # eigen / celerite / libgp also compile via the Rcpp plugin path.
    cppflags <- paste(
        paste0("-I", inc),
        paste0("-I", mcmclib_inc),
        paste0("-I", bmo_inc),
        paste0("-I", eigen_inc),
        paste0("-I", celerite_inc),
        paste0("-I", libgp_inc),
        "-DMCMC_ENABLE_ARMA_WRAPPERS",
        "-DARMA_DONT_USE_WRAPPER",
        "-DAI4BAYESCODE_RCPP_MODULE"
    )

    list(
        includes = "",
        env = list(PKG_CPPFLAGS = cppflags)
    )
}

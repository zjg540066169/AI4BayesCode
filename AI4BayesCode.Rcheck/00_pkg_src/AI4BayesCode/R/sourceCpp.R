#' Compile and load an AI4BayesCode-using sampler .cpp file (legacy alias)
#'
#' Thin back-compatibility wrapper around [ai4bayescode_source()], which is now
#' the canonical self-contained sourcer. Prefer [ai4bayescode_source()] in new
#' code; this alias is retained so existing scripts keep working.
#'
#' Earlier versions injected a `// [[Rcpp::depends(AI4BayesCode)]]` directive
#' and `#ifndef`-guarded preprocessor defines into a temporary copy of the
#' source, because the Rcpp plugin path (see [inlineCxxPlugin()]) could only
#' forward the first `-I` and dropped all `-D`. [ai4bayescode_source()] removes
#' that limitation by passing all include roots and defines as real flags via
#' `R_MAKEVARS_USER`, so no source mutation is needed and the full example set
#' (eigen / celerite / libgp) compiles.
#'
#' @param file Path to a `.cpp` file (or a source string -- forwarded to
#'   [ai4bayescode_source()], which accepts either).
#' @param env Environment in which the compiled Rcpp module class is bound.
#'   Defaults to the caller's environment.
#' @param ... Passed through to [ai4bayescode_source()] (e.g. `rebuild`,
#'   `verbose`, `extra_cppflags`, `extra_libs`).
#'
#' @return Invisibly `NULL`. The side-effect is that the Rcpp module declared
#'   in the `.cpp` becomes loadable in `env`.
#'
#' @seealso [ai4bayescode_source()] (preferred), [ai4bayescode_example()].
#'
#' @examples
#' \dontrun{
#' library(AI4BayesCode)
#'
#' # Use a shipped example
#' ai4bayescode_example("GaussianLocationScale")
#' m <- new(GaussianLocationScale, rnorm(100, 2, 1.5), seed = 1L,
#'          keep_history = TRUE)
#' m$step(4000L)
#'
#' # Use a user-generated .cpp
#' ai4bayescode_sourceCpp("./my_model.cpp")
#' }
#' @export
ai4bayescode_sourceCpp <- function(file, env = parent.frame(), ...) {
    ai4bayescode_source(file, env = env, ...)
}

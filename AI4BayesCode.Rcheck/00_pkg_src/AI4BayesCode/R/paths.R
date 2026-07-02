#' Paths to bundled AI4BayesCode assets
#'
#' These helpers return absolute paths to the skills / examples / include
#' directories shipped inside the installed AI4BayesCode package. They
#' exist so that AI coding agents (Claude Code, Cursor, etc.) and user
#' scripts can locate AI4BayesCode assets without hard-coding install
#' locations.
#'
#' @importFrom Rcpp sourceCpp
#' @importFrom RcppArmadillo RcppArmadillo.package.skeleton
#' @importFrom utils packageVersion head tail
#'
#' @param file Optional file name (relative to the directory). If
#'   `NULL`, returns the directory itself.
#' @return Absolute path as a character string. Returns `""` if the
#'   package is not installed correctly or the requested file does not
#'   exist.
#' @name ai4bayescode_paths
#'
#' @examples
#' \dontrun{
#' # Where AI4BayesCode is installed
#' ai4bayescode_include_path()
#' #> "/Library/Frameworks/R.framework/Versions/.../AI4BayesCode/include"
#'
#' # Skills directory (for AI agent context loading)
#' ai4bayescode_skills_path()
#' ai4bayescode_skills_path("start.md")
#'
#' # A specific shipped example
#' ai4bayescode_examples_path("GaussianLocationScale.cpp")
#' }
NULL

#' @rdname ai4bayescode_paths
#' @export
ai4bayescode_include_path <- function() {
    system.file("include", package = "AI4BayesCode")
}

#' @rdname ai4bayescode_paths
#' @export
ai4bayescode_skills_path <- function(file = NULL) {
    if (is.null(file)) {
        system.file("skills", package = "AI4BayesCode")
    } else if (identical(file, "start.md")) {
        # start.md is the entry-point doc at the package root, not in skills/
        system.file("start.md", package = "AI4BayesCode")
    } else {
        system.file("skills", file, package = "AI4BayesCode")
    }
}

#' @rdname ai4bayescode_paths
#' @export
ai4bayescode_examples_path <- function(file = NULL) {
    if (is.null(file)) {
        system.file("examples", package = "AI4BayesCode")
    } else {
        system.file("examples", file, package = "AI4BayesCode")
    }
}

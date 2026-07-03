#' Compile + load a built-in AI4BayesCode example by name
#'
#' Convenience wrapper: looks up `<name>.cpp` in the package's bundled
#' `inst/examples/` directory and runs [ai4bayescode_sourceCpp()] on it.
#' After the call, the corresponding Rcpp module class is loaded in the
#' calling environment and can be instantiated with
#' `new(<ClassName>, ...)`.
#'
#' @param name Character. Example name without the `.cpp` extension,
#'   e.g. `"GaussianLocationScale"`. See [ai4bayescode_list_examples()]
#'   for the full set.
#' @param env Environment in which the compiled Rcpp module class is bound.
#'   Defaults to the caller's environment.
#' @param ... Forwarded to [ai4bayescode_source()] (e.g. `rebuild`, `verbose`).
#'
#' @return Invisibly `NULL`.
#'
#' @examples
#' \dontrun{
#' library(AI4BayesCode)
#'
#' ai4bayescode_example("GaussianLocationScale")
#' m <- new(GaussianLocationScale, rnorm(100), seed = 1L, keep_history = TRUE)
#' m$step(2000L)
#'
#' ai4bayescode_example("HierarchicalLM_joint")
#' # construct + sample with HierarchicalLM_joint
#' }
#' @export
ai4bayescode_example <- function(name, env = parent.frame(), ...) {
    if (!is.character(name) || length(name) != 1L) {
        stop("`name` must be a single character string", call. = FALSE)
    }

    # Tolerate a trailing .cpp so example("Foo.cpp") works, matching Python's
    # example(). ai4bayescode_examples_path() re-appends .cpp for a bare stem.
    name <- sub("\\.cpp$", "", name)
    cpp <- ai4bayescode_examples_path(paste0(name, ".cpp"))
    if (!nzchar(cpp)) {
        avail <- ai4bayescode_list_examples()
        stop("Example '", name, "' not found.\n",
             "Available examples (", length(avail), "):\n  ",
             paste(avail, collapse = ", "),
             call. = FALSE)
    }

    ai4bayescode_sourceCpp(cpp, env = env, ...)
}

#' List all built-in AI4BayesCode examples
#'
#' Returns the names of all `.cpp` files shipped in the package's
#' `inst/examples/` directory, without the `.cpp` extension.
#'
#' @return Character vector of example names.
#'
#' @examples
#' \dontrun{
#' library(AI4BayesCode)
#' ai4bayescode_list_examples()
#' #>  [1] "ARDLasso"  "BSplineRegression"  "BartNoise"  ...
#' }
#' @export
ai4bayescode_list_examples <- function() {
    dir <- ai4bayescode_examples_path()
    if (!nzchar(dir)) return(character(0))
    files <- list.files(dir, pattern = "\\.cpp$")
    # radix sort = C-locale/byte order so the ordering matches Python's
    # sorted() (a locale sort would reorder e.g. BSplineRegression vs BartNoise).
    sort(sub("\\.cpp$", "", files), method = "radix")
}

#' List all bundled AI4BayesCode skill files
#'
#' Returns the names of all `.md` skill files shipped in the package's
#' `inst/skills/` directory. AI coding agents can read these via
#' [ai4bayescode_skills_path()] to follow the AI4BayesCode workflow
#' (see `start.md` as the canonical entry point).
#'
#' @return Character vector of skill file names (including `.md`).
#'
#' @examples
#' \dontrun{
#' library(AI4BayesCode)
#' ai4bayescode_list_skills()
#' #>  [1] "block_catalogue/index.md" "codegen.md" "codegen_cpp.md" ...
#'
#' # Read a skill file
#' cat(readLines(ai4bayescode_skills_path("start.md")), sep = "\n")
#' }
#' @export
ai4bayescode_list_skills <- function() {
    dir <- ai4bayescode_skills_path()
    mid <- if (!nzchar(dir)) character(0) else list.files(dir, pattern = "\\.md$")
    # radix sort = byte order to match Python's sorted() (list_skills).
    mid <- sort(unique(mid), method = "radix")
    # Order must match Python's list_skills() contract exactly:
    #   start.md FIRST, then the top-level skills/*.md (byte order), then
    #   block_catalogue/index.md LAST. start.md lives at the package root (not
    #   in skills/) but IS reachable via ai4bayescode_skills_path("start.md");
    #   block_catalogue is a folder (index.md + per-block cards) that the
    #   non-recursive list.files() above misses. Do NOT sort these two into the
    #   middle -- prepend / append them so the API order is stable and coherent.
    files <- mid
    if (nzchar(system.file("start.md", package = "AI4BayesCode")))
        files <- c("start.md", files)
    if (nzchar(dir) && file.exists(file.path(dir, "block_catalogue", "index.md")))
        files <- c(files, "block_catalogue/index.md")
    files
}

#' Return the installed AI4BayesCode package version
#'
#' Convenience wrapper around [utils::packageVersion()] so generated
#' `.cpp` files can stamp their provenance and so AI agents can check
#' compatibility.
#'
#' @return A `package_version` object.
#'
#' @examples
#' \dontrun{
#' library(AI4BayesCode)
#' ai4bayescode_version()
#' #>  [1] '0.9.0'
#' }
#' @export
ai4bayescode_version <- function() {
    utils::packageVersion("AI4BayesCode")
}

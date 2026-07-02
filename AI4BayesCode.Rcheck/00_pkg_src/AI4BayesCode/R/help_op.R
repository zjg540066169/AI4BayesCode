# help_op.R -- `?BartNoise` shows the same card as ai4bayescode_doc("BartNoise").

#' Block/example cards via `?`
#'
#' A thin wrapper around R's help operator: when the topic is a known
#' AI4BayesCode example/block name, `?BartNoise` prints the same card as
#' [ai4bayescode_doc()]. Any other topic is forwarded **unchanged** to
#' \code{utils::`?`}, so ordinary R help (e.g. `?lm`) is unaffected -- this only
#' *adds* block lookup on top of normal `?`.
#'
#' @param e1 Help topic. An AI4BayesCode example name (e.g. `BartNoise`) shows
#'   its card; anything else falls through to normal R help.
#' @param e2 Optional second argument for the `type?topic` form; when supplied
#'   the call is forwarded to \code{utils::`?`}.
#' @return Invisibly `NULL` after printing a block card; otherwise whatever
#'   \code{utils::`?`} returns.
#' @seealso [ai4bayescode_doc()], [ai4bayescode_list_examples()]
#' @examples
#' \dontrun{
#' ?BartNoise   # AI4BayesCode card == ai4bayescode_doc("BartNoise")
#' ?lm          # ordinary R help, unchanged
#' }
#' @usage NULL
#' @export
`?` <- function(e1, e2) {
    if (missing(e2)) {
        nm <- tryCatch({
            s <- substitute(e1)
            if (is.symbol(s)) as.character(s)
            else if (is.character(e1) && length(e1) == 1L) e1
            else NA_character_
        }, error = function(...) NA_character_)
        if (length(nm) == 1L && !is.na(nm) && nzchar(nm) &&
            (nm %in% ai4bayescode_list_examples() ||
             exists(nm, envir = .ai4b_doc_registry, inherits = FALSE))) {
            return(invisible(ai4bayescode_doc(nm)))
        }
    }
    # Not an AI4BayesCode block -> forward to normal R help, unchanged.
    cl <- sys.call()
    cl[[1L]] <- quote(utils::`?`)
    eval.parent(cl)
}

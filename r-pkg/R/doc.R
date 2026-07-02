# doc.R
#
# ai4bayescode_doc(): print a human-readable usage card for a sampler loaded
# (or about to be loaded) via ai4bayescode_source() / ai4bayescode_example().
#
# Rcpp module classes do not get an R help page, and `new(<Class>, ...)` does
# not expose the constructor argument NAMES. This helper recovers them by
# parsing the .cpp source (real ctor arg names + types + defaults, the header
# doc-block, and the RCPP_MODULE method docstrings) and falls back to runtime
# class introspection when the source is not available.

# Registry mapping an exposed class name -> the .cpp it was sourced from.
# Populated by ai4bayescode_source(); read by ai4bayescode_doc().
.ai4b_doc_registry <- new.env(parent = emptyenv())

#' @keywords internal
#' @noRd
.ai4b_doc_register <- function(cpp_file) {
    src <- tryCatch(paste(readLines(cpp_file, warn = FALSE), collapse = "\n"),
                    error = function(e) "")
    for (nm in .ai4b_exposed_class_names(src)) {
        assign(nm, normalizePath(cpp_file, mustWork = FALSE),
               envir = .ai4b_doc_registry)
    }
    invisible(.ai4b_exposed_class_names(src))
}

# Exposed class names from `Rcpp::class_<X>("Name")` or `class_<X>("Name")`.
#' @keywords internal
#' @noRd
.ai4b_exposed_class_names <- function(src) {
    m <- gregexpr('class_<[^>]+>\\s*\\(\\s*"([^"]+)"', src, perl = TRUE)
    hit <- regmatches(src, m)[[1]]
    if (!length(hit)) return(character(0))
    unique(sub('.*"([^"]+)".*', "\\1", hit))
}

# Map a C++ type to a short R-friendly hint.
#' @keywords internal
#' @noRd
.ai4b_type_hint <- function(type) {
    t <- gsub("const|&|\\s+", " ", type); t <- trimws(gsub("\\s+", " ", t))
    dplc <- function(p) grepl(p, t, perl = TRUE)
    if (dplc("arma::mat|Mat<")) return("numeric matrix")
    if (dplc("arma::ivec|arma::uvec|Col<arma::(sword|uword)")) return("integer vector")
    if (dplc("arma::vec|Col<double|NumericVector")) return("numeric vector")
    if (dplc("NumericMatrix|IntegerMatrix")) return("numeric matrix")
    if (dplc("IntegerVector")) return("integer vector")
    if (dplc("\\bbool\\b")) return("logical (TRUE/FALSE)")
    if (dplc("\\bint\\b|size_t|uint")) return("integer (e.g. 1L)")
    if (dplc("double|float")) return("numeric")
    if (dplc("std::string|CharacterVector|String")) return("character")
    trimws(type)
}

# Given the index of an opening '(' in src, balance-match its ')' and parse
# the parameter list into args (name, type, hint, default). Returns NULL on
# imbalance, list() for an empty list.
#' @keywords internal
#' @noRd
.ai4b_args_after <- function(src, open) {
    chars <- strsplit(substring(src, open), "")[[1]]
    depth <- 0L; end <- NA_integer_
    for (i in seq_along(chars)) {
        if (chars[i] == "(") depth <- depth + 1L
        else if (chars[i] == ")") { depth <- depth - 1L; if (depth == 0L) { end <- i; break } }
    }
    if (is.na(end)) return(NULL)
    inside <- gsub("\\s+", " ", trimws(substring(src, open + 1L, open + end - 2L)))
    if (!nzchar(inside)) return(list())
    parts <- character(0); buf <- ""; ang <- 0L            # split top-level commas
    for (ch in strsplit(inside, "")[[1]]) {
        if (ch == "<") ang <- ang + 1L else if (ch == ">") ang <- ang - 1L
        if (ch == "," && ang == 0L) { parts <- c(parts, buf); buf <- "" } else buf <- paste0(buf, ch)
    }
    parts <- c(parts, buf)
    lapply(parts, function(p) {
        p <- trimws(p); default <- NA_character_
        if (grepl("=", p)) { sp <- strsplit(p, "=", fixed = TRUE)[[1]]
            default <- trimws(paste(sp[-1], collapse = "=")); p <- trimws(sp[1]) }
        toks <- strsplit(p, "\\s+")[[1]]
        last <- toks[length(toks)]
        name <- gsub("[&*]", "", last)
        # Strip the trailing name token by LENGTH, not as a regex (a name with
        # regex metachars like '(' would otherwise crash sub()).
        type <- trimws(substring(p, 1L, nchar(p) - nchar(last)))
        list(name = name, type = type, hint = .ai4b_type_hint(type), default = default)
    })
}

# Parse the constructor declaration `^<indent>ClassName(...)` -> args, or NULL.
#' @keywords internal
#' @noRd
.ai4b_parse_constructor <- function(src, class_name) {
    if (is.null(class_name) || is.na(class_name) || !nzchar(class_name)) return(NULL)
    # Strip C++ comments FIRST: a generated constructor can carry inline `//`
    # notes (e.g. `double sigma,  // FIXED (exposed constructor arg, NOT sampled)`)
    # whose parens/commas would otherwise corrupt the balanced-paren arg parse.
    src <- gsub("//[^\n]*", "", src)                        # line comments
    src <- gsub("(?s)/\\*.*?\\*/", "", src, perl = TRUE)    # block comments
    loc <- regexpr(sprintf("(?m)^[ \\t]*%s[ \\t]*\\(", class_name), src, perl = TRUE)
    if (loc < 0) return(NULL)
    .ai4b_args_after(src, loc + attr(loc, "match.length") - 1L)
}

# Parse `// [[Rcpp::export]]` free functions (name -> args) for files that
# expose functions rather than an RCPP_MODULE class.
#' @keywords internal
#' @noRd
.ai4b_parse_exports <- function(src) {
    starts <- gregexpr("\\[\\[\\s*Rcpp::export[^]]*\\]\\]", src, perl = TRUE)[[1]]
    if (starts[1] < 0) return(list())
    lens <- attr(starts, "match.length"); out <- list()
    for (i in seq_along(starts)) {
        after <- substring(src, starts[i] + lens[i])       # text after the attribute
        fm <- regexpr("^\\s*[A-Za-z_][\\w:<>,&* ]*?\\b[A-Za-z_][A-Za-z0-9_]*\\s*\\(", after, perl = TRUE)
        if (fm < 0) next
        sig_head <- substring(after, 1, attr(fm, "match.length"))
        fname <- sub(".*\\b([A-Za-z_][A-Za-z0-9_]*)\\s*\\($", "\\1", sig_head)
        open_abs <- starts[i] + lens[i] - 1L + attr(fm, "match.length")   # index of '('
        args <- .ai4b_args_after(src, open_abs)
        if (!is.null(args)) out[[fname]] <- args
    }
    out
}

# Method name -> docstring from `.method("name", &C::m, "doc")`.
#' @keywords internal
#' @noRd
.ai4b_parse_methods <- function(src) {
    m <- gregexpr('\\.method\\(\\s*"([^"]+)"\\s*,\\s*&[^,\\)]+(?:,\\s*"((?:[^"\\\\]|\\\\.)*)")?',
                  src, perl = TRUE)
    hits <- regmatches(src, m)[[1]]
    out <- list()
    for (h in hits) {
        nm  <- sub('.*\\.method\\(\\s*"([^"]+)".*', "\\1", h)
        doc <- if (grepl(',\\s*"', sub('^[^,]*,[^,]*', "", h)))
                   sub('.*,\\s*"((?:[^"\\\\]|\\\\.)*)".*', "\\1", h) else ""
        out[[nm]] <- gsub('\\\\"', '"', doc)
    }
    out
}

# Top-of-file comment block, trimmed (drop copyright + rule lines).
#' @keywords internal
#' @noRd
.ai4b_parse_header <- function(src) {
    lines <- strsplit(src, "\n", fixed = TRUE)[[1]]
    out <- character(0)
    for (ln in lines) {
        s <- trimws(ln)
        if (grepl("^#", s) || grepl("^//\\s*\\[\\[", s) ||
            grepl("^//\\s*@example:", s)) break                       # code / attribute / example block
        if (grepl("^//", s)) out <- c(out, sub("^//\\s?", "", s)) else if (nzchar(s)) break
    }
    out <- out[!grepl("Copyright|Licensed under|GPL|See COPYING|^=+$|^-+$", out)]
    # keep from the first non-empty meaningful line; cap length
    out <- out[cumsum(nzchar(out)) > 0]
    head(out, 18L)
}

# Extract a language-tagged runnable example from the header doc-block:
#   // @example:R ... // @example:python ... // @example:end
# Returns the chosen language's lines (// prefix stripped), or NULL if absent.
#' @keywords internal
#' @noRd
.ai4b_parse_example <- function(src, lang = "R") {
    lines <- strsplit(src, "\n", fixed = TRUE)[[1]]
    open_re <- sprintf("^\\s*//\\s*@example:%s\\b", lang)
    st <- which(grepl(open_re, lines, perl = TRUE, ignore.case = TRUE))
    if (!length(st)) return(NULL)
    out <- character(0); i <- st[1] + 1L
    while (i <= length(lines)) {
        ln <- lines[i]
        if (grepl("^\\s*//\\s*@example:", ln, perl = TRUE)) break   # next tag or :end
        if (!grepl("^\\s*//", ln)) break                            # left the comment block
        out <- c(out, sub("^\\s*//\\s?", "", ln))
        i <- i + 1L
    }
    while (length(out) && !nzchar(trimws(out[length(out)]))) out <- out[-length(out)]
    if (!length(out)) return(NULL)
    out
}

# Friendly fallback descriptions for the canonical interface methods.
.ai4b_canonical_methods <- list(
    step         = "step(n_steps)            advance the sampler n_steps iterations",
    get_current  = "get_current()            current draw of each parameter -> named list",
    set_current  = "set_current(params)      set the current draw (named list)",
    predict_at   = "predict_at(new_data)     posterior prediction at new_data",
    get_dag      = "get_dag()                model DAG (feed to ai4bayescode_plot_dag)",
    get_history  = "get_history()            all kept draws (needs keep_history=TRUE)",
    readapt_NUTS = "readapt_NUTS(n, adapt)   re-run NUTS warm-up (online / sequential use)"
)

#' Print a usage card for an AI4BayesCode sampler
#'
#' Recovers the constructor argument names + types + defaults, the model
#' description, and the available methods for a sampler compiled by
#' [ai4bayescode_source()] / [ai4bayescode_example()], and prints a copy-paste
#' usage card. Solves the "I don't know what `new(<Class>, ...)` wants" gap that
#' Rcpp modules leave.
#'
#' @param x One of: the loaded class object (e.g. `ai4bayescode_doc(MyModel)`),
#'   the class name as a string (`"MyModel"`), or a path to the `.cpp` source.
#' @return Invisibly, a list with the parsed `class`, `constructor`, `methods`,
#'   and `description`. Called mainly for its printed side effect.
#' @examples
#' \dontrun{
#' ai4bayescode_example("GaussianLocationScale")
#' ai4bayescode_doc(GaussianLocationScale)
#' ai4bayescode_doc("./my_model.cpp")
#' }
#' @export
ai4bayescode_doc <- function(x) {
    expr <- substitute(x)          # capture BEFORE x is forced below
    class_name <- NULL; src_path <- NULL
    if (is.character(x) && length(x) == 1L && grepl("\\.cpp$", x) && file.exists(x)) {
        src_path <- normalizePath(x, mustWork = FALSE)
    } else if (is.character(x)) {
        class_name <- x
    } else {
        class_name <- if (is.symbol(expr)) as.character(expr) else deparse(expr)
    }
    if (!is.null(class_name)) {
        # the loaded instance S4 class is "Rcpp_<Name>"; the registry key is
        # "<Name>" -- try both, and display the un-prefixed name.
        for (cn in unique(c(class_name, sub("^Rcpp_", "", class_name)))) {
            if (exists(cn, envir = .ai4b_doc_registry, inherits = FALSE)) {
                class_name <- cn
                src_path   <- get(cn, envir = .ai4b_doc_registry)
                break
            }
        }
        class_name <- sub("^Rcpp_", "", class_name)
    }

    # Bundled-example fallback: if the name wasn't found in the compile registry,
    # resolve <name>.cpp from the installed examples so doc() shows the full card
    # for any shipped example -- not just classes that have been compiled.
    if (is.null(src_path) && !is.null(class_name) && nzchar(class_name)) {
        cand <- ai4bayescode_examples_path(paste0(class_name, ".cpp"))
        if (length(cand) == 1L && nzchar(cand) && file.exists(cand)) src_path <- cand
    }

    src <- if (!is.null(src_path) && file.exists(src_path))
        paste(readLines(src_path, warn = FALSE), collapse = "\n") else NULL

    if (is.null(class_name) && !is.null(src))
        class_name <- .ai4b_exposed_class_names(src)[1]

    ctor <- if (!is.null(src)) .ai4b_parse_constructor(src, class_name) else NULL
    mdoc <- if (!is.null(src)) .ai4b_parse_methods(src) else list()
    desc <- if (!is.null(src)) .ai4b_parse_header(src) else character(0)
    bar  <- strrep("\u2500", 64L)

    # No RCPP_MODULE class -> this file exposes [[Rcpp::export]] function(s).
    exports <- if (!is.null(src) && (is.null(class_name) || is.na(class_name)))
        .ai4b_parse_exports(src) else list()
    if (length(exports)) {
        cat(bar, "\n  ", basename(src_path),
            "   (exported function(s), not a sampler class)\n", bar, "\n", sep = "")
        if (length(desc)) { for (d in desc) cat("  ", d, "\n", sep = ""); cat(bar, "\n") }
        for (fn in names(exports)) {
            cat("Call:\n  ", fn, "(\n", sep = "")
            ar <- exports[[fn]]
            for (i in seq_along(ar)) {
                a <- ar[[i]]
                tl <- if (!is.na(a$default)) sprintf("  [default %s]", a$default) else ""
                cm <- if (i < length(ar)) "," else ")"
                cat(sprintf("      %-16s%s   # %s%s\n", a$name, cm, a$hint, tl))
            }
        }
        cat(bar, "\n")
        return(invisible(list(class = NA_character_, exports = exports,
                              description = desc, source = src_path)))
    }

    cat(bar, "\n")
    cat("  ", class_name, "\n", sep = "")
    if (length(desc)) { cat(bar, "\n"); for (d in desc) cat("  ", d, "\n", sep = "") }
    cat(bar, "\n")

    if (!is.null(ctor)) {
        cat("Construct:\n")
        argline <- vapply(ctor, function(a) a$name, "")
        cat("  new(", class_name, ",\n", sep = "")
        for (i in seq_along(ctor)) {
            a <- ctor[[i]]
            tail <- if (!is.na(a$default)) sprintf("  [optional, default %s]", a$default) else ""
            comma <- if (i < length(ctor)) "," else ")"
            cat(sprintf("      %-14s%s   # %s%s\n", a$name, comma, a$hint, tail))
        }
    } else {
        cat("Construct:  new(", class_name, ", ...)   ",
            "(source not found - pass the .cpp path for argument names)\n", sep = "")
    }

    cat("\nMethods:\n")
    shown <- character(0)
    for (nm in names(.ai4b_canonical_methods)) {
        registered <- is.null(src) || grepl(sprintf('\\.method\\(\\s*"%s"', nm), src)
        if (!registered && !is.null(ctor)) next   # method not in this model

        doc <- mdoc[[nm]]
        line <- if (!is.null(doc) && nzchar(doc))
            sprintf("%-24s %s", paste0(nm, "()"), doc) else .ai4b_canonical_methods[[nm]]
        cat("  $", line, "\n", sep = ""); shown <- c(shown, nm)
    }
    for (nm in setdiff(names(mdoc), shown)) {        # any custom (non-canonical) methods
        doc <- mdoc[[nm]]
        cat("  $", sprintf("%-24s %s", paste0(nm, "()"), if (nzchar(doc)) doc else ""), "\n", sep = "")
    }

    ex_block <- if (!is.null(src)) .ai4b_parse_example(src, "R") else NULL
    if (!is.null(ex_block)) {
        # Author-supplied runnable DGP from the .cpp header (@example:R).
        cat("\nExample:\n")
        for (ln in ex_block) cat(ln, "\n", sep = "")
    } else if (!is.null(ctor)) {
        # Fallback: auto-generated new(...) stub (placeholder data args).
        ex_args <- vapply(ctor, function(a) {
            if (!is.na(a$default)) NA_character_ else a$name
        }, "")
        ex_args <- ex_args[!is.na(ex_args)]
        cat("\nExample:\n")
        cat("  m <- new(", class_name, if (length(ex_args)) paste0(", ", paste(ex_args, collapse = ", ")) else "", ")\n", sep = "")
        cat("  m$step(2000); str(m$get_current())\n")
    }
    cat(bar, "\n")

    invisible(list(class = class_name, constructor = ctor,
                   methods = mdoc, description = desc, example = ex_block,
                   source = src_path))
}

# named_ctor.R
#
# Named-argument constructors for Rcpp module classes.
#
# Rcpp dispatches a module constructor on ARITY ALONE -- its default validator
# is `yes_arity<N>`, whose whole body is `return nargs == n`. Argument NAMES
# never reach the dispatcher, and C++ default arguments are invisible to R.
# So for a stock module class `new(Model, X = X, y = y, rng_seed = 7L)` does
# not mean what it reads as: the three values are passed POSITIONALLY into
# whichever constructor happens to have arity 3, and the names are dropped in
# silence.
#
# The names are still intact one level up, in the reference-class `initialize`
# method Rcpp installs on each exposed class:
#
#     if (nargs()) Rcpp::cpp_object_initializer(.self, .refClassDef, ...)
#     else         Rcpp::cpp_object_dummy(.self, .refClassDef)
#
# This file replaces that method, per class, with one that matches arguments
# R-style (exact then unique-partial names, then positional fill), supplies
# the C++ default for anything omitted, rejects unknown or duplicated names,
# and only then calls the original initializer with a COMPLETE positional
# list. Nothing in Rcpp is patched globally; only classes loaded through
# ai4bayescode_source() are affected.

# Convert a C++ default-argument literal to an R value, guided by the declared
# type. Returns a one-element list on success and NULL when the literal is not
# something R can reconstruct (in which case the argument stays REQUIRED --
# guessing a value the user did not write is the failure this file exists to
# prevent).
#' @keywords internal
#' @noRd
.ai4b_cpp_default_to_r <- function(default, type) {
    if (is.null(default) || is.na(default)) return(NULL)
    d <- trimws(default)
    d <- sub("^\\{(.*)\\}$", "\\1", d)          # brace-init: `= {3}`
    d <- trimws(sub("[uUlLfF]+$", "", d))       # literal suffixes: 3u, 1.0f
    if (!nzchar(d)) return(NULL)
    if (identical(d, "true"))  return(list(TRUE))
    if (identical(d, "false")) return(list(FALSE))
    # A default-constructed armadillo container is EMPTY, and an empty R
    # vector/matrix converts back to exactly that -- so this default is
    # reconstructible, unlike an arbitrary expression.
    if (grepl("^arma::(vec|colvec|rowvec|uvec|ivec)\\(\\)$", d))
        return(list(numeric(0)))
    if (grepl("^arma::(mat|umat|imat)\\(\\)$", d))
        return(list(matrix(numeric(0), 0L, 0L)))
    if (!grepl("^[+-]?([0-9]+\\.?[0-9]*|\\.[0-9]+)([eE][+-]?[0-9]+)?$", d))
        return(NULL)
    hint <- .ai4b_type_hint(type)
    if (identical(hint, "logical (TRUE/FALSE)")) return(list(as.numeric(d) != 0))
    if (identical(hint, "integer (e.g. 1L)"))    return(list(as.integer(as.numeric(d))))
    list(as.numeric(d))
}

# Canonical form of a C++ type, so a declared parameter type and the type
# Rcpp reports for a bound constructor compare equal.
#' @keywords internal
#' @noRd
.ai4b_norm_type <- function(t) {
    # The .cpp says `const arma::vec&`; Rcpp reports the resolved
    # `arma::Col<double>` for the same parameter. Comparing the spellings
    # directly would never match, so both sides go through the same coarse
    # type hint, which maps every spelling of a kind onto one label.
    vapply(t, function(x) .ai4b_type_hint(gsub("[&*]", "", x)),
           character(1), USE.NAMES = FALSE)
}

# Every constructor a class declares, in declaration order. A class may
# declare more than one -- the mixtures pair a simple data-driven constructor
# with an advanced one taking explicit hyperparameters -- and which of them a
# call means is decided by the arguments themselves, not up front.
#' @keywords internal
#' @noRd
.ai4b_parse_all_constructors <- function(src, class_name) {
    if (is.null(class_name) || is.na(class_name) || !nzchar(class_name))
        return(list())
    src <- gsub("//[^\n]*", "", src)
    src <- gsub("(?s)/\\*.*?\\*/", "", src, perl = TRUE)
    locs <- gregexpr(sprintf("(?m)^[ \\t]*%s[ \\t]*\\(", class_name), src,
                     perl = TRUE)[[1]]
    if (locs[1L] < 0) return(list())
    lens <- attr(locs, "match.length")
    out <- list()
    for (i in seq_along(locs)) {
        a <- .ai4b_args_after(src, locs[i] + lens[i] - 1L)
        if (!is.null(a) && length(a)) out[[length(out) + 1L]] <- a
    }
    out
}

# Constructor signatures for one class: for each declared constructor, the
# argument names, canonical types, R-side defaults, and a flag per argument
# saying whether a usable default exists.
#' @keywords internal
#' @noRd
.ai4b_ctor_signatures <- function(cpp_file, class_name) {
    src <- tryCatch(paste(readLines(cpp_file, warn = FALSE), collapse = "\n"),
                    error = function(e) NULL)
    if (is.null(src)) return(list())
    all <- tryCatch(.ai4b_parse_all_constructors(src, class_name),
                    error = function(e) list())
    out <- list()
    for (args in all) {
        nms <- vapply(args, function(a) a$name, character(1))
        if (anyNA(nms) || !all(nzchar(nms)) || anyDuplicated(nms)) next
        defs <- vector("list", length(args)); has <- logical(length(args))
        for (i in seq_along(args)) {
            v <- .ai4b_cpp_default_to_r(args[[i]]$default, args[[i]]$type)
            if (!is.null(v)) { defs[[i]] <- v[[1L]]; has[i] <- TRUE }
        }
        out[[length(out) + 1L]] <- list(
            names = nms,
            types = vapply(args, function(a) .ai4b_norm_type(a$type), character(1)),
            defaults = defs, has_default = has)
    }
    out
}

# The constructors Rcpp actually bound, as arity plus canonical type list,
# read from each C++Constructor's reported signature.
#' @keywords internal
#' @noRd
.ai4b_bound_ctors <- function(gen_holder) {
    ks <- tryCatch(gen_holder@constructors, error = function(e) list())
    out <- list()
    for (k in ks) {
        n <- tryCatch(as.integer(k$nargs), error = function(e) NA_integer_)
        sg <- tryCatch(as.character(k$signature), error = function(e) NA_character_)
        ty <- character(0)
        if (!is.na(sg)) {
            inner <- sub("^[^(]*\\((.*)\\)\\s*$", "\\1", sg)
            if (nzchar(trimws(inner)))
                ty <- .ai4b_norm_type(.ai4b_split_types(inner))
        }
        if (!is.na(n)) out[[length(out) + 1L]] <- list(nargs = n, types = ty)
    }
    out
}

# Split a comma-separated C++ parameter/type list at TOP level only. A
# templated type carries commas of its own (a map, a nested vector), and
# splitting on every comma would report more types than the constructor has
# parameters -- which silently disqualified the class from named arguments.
#' @keywords internal
#' @noRd
.ai4b_split_types <- function(s) {
    chars <- strsplit(s, "")[[1]]
    out <- character(0); buf <- ""; depth <- 0L
    for (ch in chars) {
        if (ch == "<" || ch == "(") depth <- depth + 1L
        else if (ch == ">" || ch == ")") depth <- depth - 1L
        if (ch == "," && depth == 0L) { out <- c(out, buf); buf <- "" }
        else buf <- paste0(buf, ch)
    }
    trimws(c(out, buf))
}

# Match a call's arguments against a constructor signature, R-style. Returns
# list(values, filled); `filled` marks which slots the CALLER supplied, so the
# installer can pick the narrowest bound arity that still covers them.
#' @keywords internal
#' @noRd
.ai4b_match_ctor_args <- function(nms, defaults, has_default, args, class_name) {
    n  <- length(nms)
    an <- names(args); if (is.null(an)) an <- rep("", length(args))
    values <- vector("list", n); filled <- logical(n)

    # Assigning NULL through [[<- DELETES the element and shifts every later
    # one down, which would move already-matched values onto the wrong
    # parameters. Single-bracket assignment of a one-element list stores NULL
    # in place.
    put <- function(j, v) { values[j] <<- list(v); filled[j] <<- TRUE }

    named <- which(nzchar(an))
    # Pass 1: exact names, ALL of them, before any partial matching -- this is
    # base R's order, and doing it per-argument instead would let a partial
    # match claim a formal that a later exact name owns.
    partial <- integer(0)
    for (i in named) {
        j <- match(an[i], nms)
        if (is.na(j)) { partial <- c(partial, i); next }
        if (filled[j])
            stop(sprintf("%s: argument \"%s\" supplied more than once.",
                         class_name, nms[j]), call. = FALSE)
        put(j, args[[i]])
    }
    # Pass 2: unique partial matches, considered only against the formals that
    # pass 1 left unclaimed.
    for (i in partial) {
        cand <- which(!filled & startsWith(nms, an[i]))
        if (length(cand) > 1L)
            stop(sprintf("%s: argument \"%s\" matches several arguments: %s.",
                         class_name, an[i], paste(nms[cand], collapse = ", ")),
                 call. = FALSE)
        if (!length(cand))
            stop(sprintf("%s: unknown argument \"%s\". Valid arguments: %s.",
                         class_name, an[i], paste(nms, collapse = ", ")),
                 call. = FALSE)
        put(cand, args[[i]])
    }
    # Pass 3: positional arguments fill the slots still open, left to right.
    k <- 1L
    for (i in which(!nzchar(an))) {
        while (k <= n && filled[k]) k <- k + 1L
        if (k > n)
            stop(sprintf("%s: too many arguments; the constructor takes %d (%s).",
                         class_name, n, paste(nms, collapse = ", ")), call. = FALSE)
        put(k, args[[i]])
    }
    missing <- which(!filled & !has_default)
    if (length(missing))
        stop(sprintf("%s: missing required argument(s): %s.",
                     class_name, paste(nms[missing], collapse = ", ")),
             call. = FALSE)
    for (j in which(!filled)) values[j] <- list(defaults[[j]])
    list(values = values, filled = filled)
}

# Resolve one call against a class's constructors. Each declared constructor
# is tried in declaration order and the first that ACCEPTS the arguments wins
# -- argument names disambiguate on their own, so a call naming an advanced
# hyperparameter cannot land on the simple constructor and vice versa.
#
# Having chosen a constructor, pick which bound registration to call: the
# narrowest one whose type list is a PREFIX of that constructor's parameter
# types and that still covers every argument the caller supplied. Matching on
# types, not on arity alone, is what keeps a call off a same-arity
# registration belonging to a DIFFERENT constructor.
#' @keywords internal
#' @noRd
.ai4b_resolve_ctor_call <- function(sigs, bound, args, class_name) {
    # When a class declares several constructors and none accepts the call,
    # the useful complaint comes from the one the caller evidently MEANT --
    # the signature recognising the most of the names they supplied -- not
    # from whichever happens to be declared first.
    supplied <- names(args); if (is.null(supplied)) supplied <- character(0)
    supplied <- supplied[nzchar(supplied)]
    score <- vapply(sigs, function(sg)
        sum(vapply(supplied, function(nm)
            nm %in% sg$names || length(which(startsWith(sg$names, nm))) == 1L,
            logical(1))), integer(1))
    best_err <- NULL; best_score <- -1L
    keep_err <- function(e, i) {
        if (score[i] > best_score) { best_err <<- e; best_score <<- score[i] }
    }
    for (si in seq_along(sigs)) {
        sig <- sigs[[si]]
        m <- tryCatch(
            .ai4b_match_ctor_args(sig$names, sig$defaults, sig$has_default,
                                  args, class_name),
            error = function(e) e)
        if (inherits(m, "error")) { keep_err(m, si); next }
        need <- if (any(m$filled)) max(which(m$filled)) else 0L
        best <- NA_integer_
        for (b in bound) {
            if (b$nargs < need || b$nargs > length(sig$names)) next
            if (length(b$types) != b$nargs) next
            if (!identical(b$types, unname(sig$types[seq_len(b$nargs)]))) next
            if (is.na(best) || b$nargs < best) best <- b$nargs
        }
        if (!is.na(best)) return(m$values[seq_len(best)])
        keep_err(simpleError(sprintf(
                paste0("%s: no bound constructor accepts %d argument(s). ",
                       "Bound arities: %s."),
                class_name, need,
                paste(sort(vapply(bound, function(b) b$nargs, integer(1))),
                      collapse = ", "))), si)
    }
    stop(if (is.null(best_err))
             simpleError(sprintf("%s: no constructor matches this call.", class_name))
         else best_err)
}

# Replace one exposed class's reference-class `initialize` with the
# name-matching version. Silently does nothing when the signatures cannot be
# recovered -- the class then keeps stock Rcpp behaviour rather than acquiring
# a half-informed one.
#' @keywords internal
#' @noRd
.ai4b_install_named_ctor <- function(class_name, cpp_file, env) {
    holder <- tryCatch(get(class_name, envir = env), error = function(e) NULL)
    if (is.null(holder) || !methods::is(holder, "C++Class")) return(FALSE)
    gen <- tryCatch(holder@generator, error = function(e) NULL)
    if (is.null(gen)) return(FALSE)

    sigs <- .ai4b_ctor_signatures(cpp_file, class_name)
    if (!length(sigs)) return(FALSE)
    bound <- .ai4b_bound_ctors(holder)
    if (!length(bound)) return(FALSE)
    # Every bound registration must be reachable from some declared
    # constructor; if one is not, the parse disagrees with the module and the
    # class is left alone rather than served a guess.
    reachable <- vapply(bound, function(b) {
        any(vapply(sigs, function(sg)
            b$nargs <= length(sg$names) && length(b$types) == b$nargs &&
            identical(b$types, unname(sg$types[seq_len(b$nargs)])), logical(1)))
    }, logical(1))
    if (!all(reachable)) return(FALSE)

    # The replacement runs as a reference-class method, which is re-environed
    # onto the object, so it cannot close over anything here. Everything it
    # needs is baked into the body as a literal instead.
    f <- function(...) NULL
    body(f) <- bquote({
        if (!nargs()) return(Rcpp::cpp_object_dummy(.self, .refClassDef))
        .args <- list(...)
        # C++ handing an already-constructed object back to R
        # (Rcpp::cpp_object_maker) must pass through untouched.
        if (".object_pointer" %in% names(.args))
            return(Rcpp::cpp_object_initializer(.self, .refClassDef, ...))
        do.call(Rcpp::cpp_object_initializer,
                c(list(.self, .refClassDef),
                  .(.ai4b_resolve_ctor_call)(.(sigs), .(bound), .args,
                                             .(class_name))))
    })
    tryCatch({ gen$methods(initialize = f); TRUE }, error = function(e) FALSE)
}

# Install on every class the file exposes. Returns the names it handled.
#' @keywords internal
#' @noRd
.ai4b_install_named_ctors <- function(cpp_file, env) {
    src <- tryCatch(paste(readLines(cpp_file, warn = FALSE), collapse = "\n"),
                    error = function(e) "")
    done <- character(0)
    for (nm in .ai4b_exposed_class_names(src))
        if (isTRUE(.ai4b_install_named_ctor(nm, cpp_file, env))) done <- c(done, nm)
    done
}

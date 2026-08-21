#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# Every model with a NUTS kernel must expose readapt_NUTS. No exception.
#
# The rule is validator.md Check #23(0). It is checked HERE rather than left to
# the validator's prose because the failure it guards is self-concealing: an
# earlier wording let the trigger be read off the method list, so a wrapper
# that contained a NUTS kernel and forgot to bind readapt_NUTS was reported
# N/A -- the omission cancelled its own detection.
#
# Both arities must be bound as separate overloads with explicit casts,
# (int, bool, int) and (int, bool, int, double): Rcpp modules dispatch on
# arity and ignore C++ default arguments, so a single binding leaves one of
# the documented call shapes unreachable.
# ---------------------------------------------------------------------------
script_dir <- local({
    a <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
    if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else getwd()
})
root <- normalizePath(file.path(script_dir, ".."))

files <- list.files(file.path(root, "examples"), pattern = "\\.cpp$", full.names = TRUE)
n_nuts <- 0L; bad <- character(0); bad_arity <- character(0)

for (f in files) {
    src <- paste(readLines(f, warn = FALSE), collapse = "\n")
    # INSTANTIATION, not a mention in a comment: a config object or a
    # make_unique of the block type.
    if (!grepl("make_unique<(AI4BayesCode::)?(joint_)?nuts_block>|(joint_)?nuts_block_config [a-z]",
               src, perl = TRUE)) next
    n_nuts <- n_nuts + 1L
    nm <- sub("\\.cpp$", "", basename(f))
    # One thing to look for: the macro. Before AI4BAYESCODE_BIND_READAPT_NUTS
    # existed, each model hand-wrote the two .method() overloads and this check
    # had to parse them -- and the first version of it could not fail, because
    # it grepped the whole file and matched the method DECLARATION rather than
    # the binding. The macro binds both arities in one place, so "did you call
    # it" is the whole question.
    if (!grepl("AI4BAYESCODE_BIND_READAPT_NUTS", src, fixed = TRUE)) {
        if (grepl('method\\("readapt_NUTS"', src))
            bad_arity <- c(bad_arity, sprintf("%s (hand-written binding; use the macro)", nm))
        else
            bad <- c(bad, nm)
    }
}

cat(sprintf("examples with a NUTS kernel: %d\n", n_nuts))
if (length(bad))
    cat(sprintf("\nFAIL: %d model(s) instantiate a NUTS kernel and do NOT bind readapt_NUTS:\n  %s\n",
                length(bad), paste(bad, collapse = "\n  ")))
if (length(bad_arity))
    cat(sprintf("\nFAIL: %d model(s) bind readapt_NUTS with only one arity:\n  %s\n",
                length(bad_arity), paste(bad_arity, collapse = "\n  ")))
if (!length(bad) && !length(bad_arity))
    cat("all of them bind readapt_NUTS, both arities: OK\n")
if (length(bad) || length(bad_arity)) quit(status = 1L)

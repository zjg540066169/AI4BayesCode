# Run the runnable `// @example:R` block of every shipped example, each in a
# fresh Rscript process, and report which ones fail. The Windows CI job uses
# this to check that generated-style samplers compile and run end to end.
#
# Usage: Rscript tools/ci/run_examples.R [part] [n_parts] [log_dir] [timeout_s]
# The examples are split round-robin into n_parts shards; this run takes one.

args    <- commandArgs(trailingOnly = TRUE)
part    <- if (length(args) >= 1L) as.integer(args[[1]]) else 1L
n_parts <- if (length(args) >= 2L) as.integer(args[[2]]) else 1L
log_dir <- if (length(args) >= 3L) args[[3]] else file.path(tempdir(), "example-logs")
timeout <- if (length(args) >= 4L) as.numeric(args[[4]]) else 1800
dir.create(log_dir, recursive = TRUE, showWarnings = FALSE)
log_dir <- normalizePath(log_dir, winslash = "/")

suppressPackageStartupMessages(library(AI4BayesCode))
all_ex <- ai4bayescode_list_examples()
mine   <- all_ex[(seq_along(all_ex) - 1L) %% n_parts == (part - 1L)]
rscript <- file.path(R.home("bin"),
                     if (.Platform$OS.type == "windows") "Rscript.exe" else "Rscript")

cat(sprintf("R %s on %s | part %d of %d | %d examples\n",
            getRversion(), R.version$platform, part, n_parts, length(mine)))

res <- data.frame(example = mine, status = "", seconds = NA_real_,
                  stringsAsFactors = FALSE)
for (i in seq_along(mine)) {
    nm   <- mine[[i]]
    cpp  <- ai4bayescode_examples_path(paste0(nm, ".cpp"))
    src  <- paste(readLines(cpp, warn = FALSE, encoding = "UTF-8"), collapse = "\n")
    code <- AI4BayesCode:::.ai4b_parse_example(src, "R")
    if (is.null(code)) { res$status[i] <- "NO-EXAMPLE"; next }

    # Each example runs in its own working directory so plot files and
    # caches never collide, and in its own process so a crash is contained.
    wd <- file.path(log_dir, paste0(nm, "_wd"))
    dir.create(wd, showWarnings = FALSE)
    script <- file.path(log_dir, paste0(nm, ".R"))
    writeLines(c(sprintf("setwd(%s)", deparse(wd)), code), script, useBytes = TRUE)

    out <- file.path(log_dir, paste0(nm, ".out.txt"))
    err <- file.path(log_dir, paste0(nm, ".err.txt"))
    t0  <- Sys.time()
    st  <- suppressWarnings(system2(rscript, c("--vanilla", shQuote(script)),
                                    stdout = out, stderr = err, timeout = timeout))
    res$seconds[i] <- round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1)
    res$status[i]  <- if (identical(as.integer(st), 0L)) "PASS"
                      else if (identical(as.integer(st), 124L)) "TIMEOUT"
                      else sprintf("FAIL(%s)", st)
    cat(sprintf("%-34s %-10s %7.1fs\n", nm, res$status[i], res$seconds[i]))
    if (res$status[i] != "PASS") {
        tail_err <- tail(readLines(err, warn = FALSE), 25L)
        cat(paste0("    | ", tail_err), sep = "\n")
    }
    flush.console()
}

write.csv(res, file.path(log_dir, "summary.csv"), row.names = FALSE)
n_bad <- sum(res$status != "PASS")
cat(sprintf("\n%d of %d examples passed\n", nrow(res) - n_bad, nrow(res)))
quit(status = if (n_bad > 0L) 1L else 0L)

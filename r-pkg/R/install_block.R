# install_block.R -- CRAN-style installation of contributed blocks from the hub
# registry. The registry (github.com/zjg540066169/AI4BayesCode-hub/registry/) is
# the "CRAN repo": each subfolder is a reviewed, pre-validated bundle
# (manifest.dcf + <block>.hpp + skills/ + test_<block>.cpp + optional examples/ +
# vendor/). install_block downloads a bundle into a PER-USER block library that
# ai4bayescode_sourceCpp() adds to the compile -I path, so #include "<block>.hpp"
# (and its vendored headers) resolve -- the local() tier in contrib.md, surfaced
# to users as a CRAN-like install.packages() family.

.AI4B_HUB_REPO <- "zjg540066169/AI4BayesCode-hub"
.AI4B_HUB_REF  <- "main"

# ---- per-user block library --------------------------------------------------

#' @keywords internal
#' @noRd
.ai4b_blocks_dir <- function() {
    file.path(tools::R_user_dir("AI4BayesCode", "data"), "blocks")
}

#' Path to the per-user contributed-block library
#'
#' The directory where [ai4bayescode_install_block()] installs blocks (under
#' `tools::R_user_dir("AI4BayesCode", "data")/blocks/`). `ai4bayescode_sourceCpp()`
#' adds every installed block (and its vendored dependencies) to the compile
#' `-I` path automatically.
#'
#' @param name Optional block name; `NULL` (default) returns the library root.
#' @return The absolute path (character).
#' @export
ai4bayescode_blocks_path <- function(name = NULL) {
    d <- .ai4b_blocks_dir()
    if (is.null(name)) d else file.path(d, name)
}

# -I flags for every installed block: the block dir (for "<block>.hpp") plus each
# vendored dependency dir. Folded into the compile flags by AI4BayesCode_sourceCpp.
#' @keywords internal
#' @noRd
.ai4b_block_cppflags <- function() {
    bdir <- .ai4b_blocks_dir()
    if (!dir.exists(bdir)) return(character())
    flags <- character()
    for (b in list.dirs(bdir, recursive = FALSE)) {
        flags <- c(flags, paste0("-I", shQuote(b)))
        vdir <- file.path(b, "vendor")
        if (dir.exists(vdir))
            for (v in list.dirs(vdir, recursive = FALSE))
                flags <- c(flags, paste0("-I", shQuote(v)))
    }
    flags
}

# ---- registry access ---------------------------------------------------------

.ai4b_raw_url  <- function(path)
    sprintf("https://raw.githubusercontent.com/%s/%s/%s", .AI4B_HUB_REPO, .AI4B_HUB_REF, path)
.ai4b_tree_url <- function()
    sprintf("https://api.github.com/repos/%s/git/trees/%s?recursive=1", .AI4B_HUB_REPO, .AI4B_HUB_REF)

# the repo file tree (data.frame: path / type / ...); needs jsonlite
#' @keywords internal
#' @noRd
.ai4b_hub_tree <- function() {
    if (!requireNamespace("jsonlite", quietly = TRUE))
        stop("install_block needs the 'jsonlite' package. install.packages('jsonlite')",
             call. = FALSE)
    tr <- tryCatch(jsonlite::fromJSON(.ai4b_tree_url()),
                   error = function(e)
                       stop("Could not reach the AI4BayesCode block registry: ",
                            conditionMessage(e), call. = FALSE))
    if (isTRUE(tr$truncated))
        warning("registry listing was truncated by GitHub; some blocks may be hidden.",
                call. = FALSE)
    tr$tree
}

# core block names are reserved (a contrib block may not shadow one)
#' @keywords internal
#' @noRd
.ai4b_core_block_names <- function() {
    inc <- file.path(ai4bayescode_include_path(), "AI4BayesCode")
    if (!nzchar(inc) || !dir.exists(inc)) return(character())
    sub("\\.hpp$", "", list.files(inc, pattern = "_block\\.hpp$"))
}

# parse "core (>= 1.0)" and compare to the installed package version
#' @keywords internal
#' @noRd
.ai4b_check_core_dep <- function(depends, name) {
    if (is.na(depends) || !nzchar(depends)) return(invisible())
    m <- regmatches(depends, regexec("core\\s*\\(\\s*>=\\s*([0-9.]+)\\s*\\)", depends))[[1]]
    if (length(m) < 2L) return(invisible())
    need <- m[2]
    have <- tryCatch(as.character(utils::packageVersion("AI4BayesCode")),
                     error = function(e) NA_character_)
    if (!is.na(have) && utils::compareVersion(have, need) < 0L)
        warning("Block '", name, "' wants AI4BayesCode core (>= ", need,
                ") but you have ", have, "; it may not compile.", call. = FALSE)
    invisible()
}

# ---- public CRAN-family API --------------------------------------------------

#' Blocks available in the hub registry (like `available.packages()`)
#'
#' @return Character vector of installable block names.
#' @export
ai4bayescode_available_blocks <- function() {
    tree  <- .ai4b_hub_tree()
    dirs  <- tree$path[tree$type == "tree"]
    hits  <- grep("^registry/[^/]+$", dirs, value = TRUE)
    sort(sub("^registry/", "", hits))
}

#' Contributed blocks already installed in the user library (like `installed.packages()`)
#'
#' @return Character vector of installed block names.
#' @export
ai4bayescode_installed_blocks <- function() {
    bdir <- .ai4b_blocks_dir()
    if (!dir.exists(bdir)) return(character(0))
    sort(basename(list.dirs(bdir, recursive = FALSE)))
}

#' Install a contributed block from the hub registry (like `install.packages()`)
#'
#' Downloads a reviewed block bundle from the AI4BayesCode hub registry into the
#' per-user block library ([ai4bayescode_blocks_path()]). After install, the block
#' header and its vendored dependencies are on `ai4bayescode_sourceCpp()`'s
#' include path and the block's skill is discoverable to codegen. Mirrors the CRAN
#' `install.packages()` model: the registry is curated + pre-validated, so install
#' is a download + manifest/version check (no compile-time sandbox).
#'
#' @param name Block name, e.g. `"nngp_gaussian_gibbs_block"`.
#' @param force Reinstall even if already present (default `FALSE`).
#' @param quiet Suppress progress + the install summary (default `FALSE`).
#' @return Invisibly, the install path.
#' @examples
#' \dontrun{
#' ai4bayescode_available_blocks()
#' ai4bayescode_install_block("nngp_gaussian_gibbs_block")
#' }
#' @export
ai4bayescode_install_block <- function(name, force = FALSE, quiet = FALSE) {
    stopifnot(is.character(name), length(name) == 1L, nzchar(name))
    dest <- ai4bayescode_blocks_path(name)
    if (dir.exists(dest) && !force) {
        if (!quiet)
            message("Block '", name, "' is already installed (", dest, ").\n",
                    "  Use force = TRUE to reinstall, or ai4bayescode_remove_block(\"",
                    name, "\").")
        return(invisible(dest))
    }
    # 1. manifest (also the existence check)
    man_txt <- tryCatch(readLines(.ai4b_raw_url(sprintf("registry/%s/manifest.dcf", name)),
                                  warn = FALSE),
                        error = function(e) NULL)
    if (is.null(man_txt)) {
        avail <- tryCatch(ai4bayescode_available_blocks(), error = function(e) character())
        stop("Block '", name, "' is not in the registry.",
             if (length(avail))
                 paste0("\nAvailable (", length(avail), "): ", paste(avail, collapse = ", "))
             else "", call. = FALSE)
    }
    man <- read.dcf(textConnection(paste(man_txt, collapse = "\n")))
    # 2. reserved core name
    if (name %in% .ai4b_core_block_names())
        stop("'", name, "' is a reserved core block name.", call. = FALSE)
    # 3. core-version dependency
    .ai4b_check_core_dep(if ("Depends" %in% colnames(man)) man[1, "Depends"] else NA, name)
    # 4. download every file in registry/<name>/
    if (!quiet) message("Installing '", name, "' from the AI4BayesCode registry ...")
    tree   <- .ai4b_hub_tree()
    prefix <- sprintf("registry/%s/", name)
    files  <- tree$path[tree$type == "blob" & startsWith(tree$path, prefix)]
    if (!length(files)) stop("Registry bundle for '", name, "' is empty.", call. = FALSE)
    tmp <- file.path(tempdir(), paste0("ai4b_blk_", name, "_", Sys.getpid()))
    unlink(tmp, recursive = TRUE); dir.create(tmp, recursive = TRUE)
    on.exit(unlink(tmp, recursive = TRUE), add = TRUE)
    for (f in files) {
        out <- file.path(tmp, sub(prefix, "", f, fixed = TRUE))
        dir.create(dirname(out), showWarnings = FALSE, recursive = TRUE)
        ok <- tryCatch({
            utils::download.file(.ai4b_raw_url(f), out, quiet = TRUE, mode = "wb"); TRUE
        }, error = function(e) FALSE)
        if (!ok || !file.exists(out)) stop("Failed to download ", f, call. = FALSE)
    }
    # 5. swap into the library atomically
    dir.create(dirname(dest), showWarnings = FALSE, recursive = TRUE)
    unlink(dest, recursive = TRUE)
    if (!file.rename(tmp, dest)) {
        file.copy(tmp, dirname(dest), recursive = TRUE)
        file.rename(file.path(dirname(dest), basename(tmp)), dest)
    }
    if (!quiet) .ai4b_block_install_report(man, dest)
    invisible(dest)
}

#' Remove an installed contributed block (like `remove.packages()`)
#'
#' @param name Block name.
#' @return Invisibly `TRUE` if removed, `FALSE` if it was not installed.
#' @export
ai4bayescode_remove_block <- function(name) {
    dest <- ai4bayescode_blocks_path(name)
    if (!dir.exists(dest)) {
        message("Block '", name, "' is not installed.")
        return(invisible(FALSE))
    }
    unlink(dest, recursive = TRUE)
    message("Removed block '", name, "'.")
    invisible(TRUE)
}

# ---- install summary (the install.packages-style message) --------------------

#' @keywords internal
#' @noRd
.ai4b_block_install_report <- function(man, dest) {
    f  <- function(k) if (k %in% colnames(man)) man[1, k] else NA_character_
    line <- function(label, v) if (!is.na(v) && nzchar(v)) message("  ", label, ": ", v)
    message("Installed block '", f("Block"), "':")
    line("version",  f("Version"))
    line("title",    f("Title"))
    line("license",  f("License"))
    line("vendored", f("Vendored"))
    message("  location: ", dest)
    ex <- f("Example")
    if (!is.na(ex) && nzchar(ex))
        message("  example:  see ", file.path(dest, ex),
                "  (compile with ai4bayescode_sourceCpp())")
    message("  -> usable now: the header + vendored deps are on the sourceCpp include path.")
    invisible()
}

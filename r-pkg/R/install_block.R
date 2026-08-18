# install_block.R -- CRAN-style installation of contributed blocks from the hub
# registry. The registry (github.com/zjg540066169/AI4BayesCode-hub/registry/) is
# the "CRAN repo": each subfolder is a reviewed, pre-validated bundle
# (manifest.dcf + <block>.hpp + skills/ + test_<block>.cpp + optional examples/ +
# vendor/). install_block downloads a bundle into a PER-USER block library that
# ai4bayescode_source() adds to the compile -I path, so #include "<block>.hpp"
# (and its vendored headers) resolve -- the local() tier in contrib.md, surfaced
# to users as a CRAN-like install.packages() family.

.AI4B_HUB_REPO <- "zjg540066169/AI4BayesCode-hub"
.AI4B_HUB_REF  <- "main"

# ---- per-user block library --------------------------------------------------

#' @keywords internal
#' @noRd
.ai4b_blocks_dir <- function() {
    # User-global, language-agnostic block store SHARED with the Python package
    # (same `$AI4BAYESCODE_DATA_HOME` override). One directory across R / Python /
    # C++ and across all projects, so a block installed once is found everywhere.
    base <- Sys.getenv("AI4BAYESCODE_DATA_HOME",
                       unset = file.path(path.expand("~"), ".AI4BayesCode"))
    file.path(base, "blocks_download")
}

#' Path to the per-user contributed-block library
#'
#' The directory where [ai4bayescode_install_block()] installs blocks: the
#' user-global, language-agnostic store `~/.AI4BayesCode/blocks_download/`
#' (shared with the Python package; override the root with the
#' `AI4BAYESCODE_DATA_HOME` environment variable). `ai4bayescode_source()`
#' adds every installed block (and its vendored dependencies) to the compile
#' `-I` path automatically.
#'
#' @param name Optional block name; `NULL` (default) returns the library root.
#' @return The absolute path (character).
#' @export
ai4bayescode_blocks_path <- function(name = NULL) {
    d <- .ai4b_blocks_dir()
    if (is.null(name)) d else file.path(d, .ai4b_check_block_name(name))
}

# A block name is ONE directory name under the block library -- never a path,
# never empty. file.path(d, "") returns the library root with a trailing
# separator, which passes dir.exists() and made remove_block("") delete every
# installed block; "../x" escapes the library altogether. Validated here, at
# the single point install / remove / lookup all resolve through.
#' @keywords internal
#' @noRd
.ai4b_check_block_name <- function(name) {
    if (!is.character(name) || length(name) != 1L || is.na(name) ||
        !nzchar(name)) {
        stop("block name must be a single non-empty, non-NA character string.",
             call. = FALSE)
    }
    if (grepl("[/\\\\]", name) || name %in% c(".", "..")) {
        stop("block name must be a plain block name, not a path: '", name,
             "'. Run ai4bayescode_installed_blocks() to see what is installed.",
             call. = FALSE)
    }
    invisible(name)
}

# Content fingerprint of every contributed-block directory that ends up on the
# -I path. The flags alone are NOT enough to key a compile cache on: reinstalling
# a block (ai4bayescode_install_block(force = TRUE) fetching a new version, or a
# hand-edit under ./blocks_local/) leaves the -I paths byte-identical, so
# Rcpp::sourceCpp's in-session cache handed back the object built from the
# PREVIOUS header. A user who updated a block and re-ran kept executing the old
# sampler -- the silent-wrong-posterior failure the contrib design exists to
# prevent. Fingerprints (relative path, size, mtime) -- a stat per file, no reads.
#' @keywords internal
#' @noRd
.ai4b_block_fingerprint <- function() {
    dirs <- c(file.path(getwd(), "blocks_local"), .ai4b_blocks_dir())
    dirs <- dirs[dir.exists(dirs)]
    if (!length(dirs)) return("")
    parts <- character(0)
    for (d in dirs) {
        fs <- list.files(d, recursive = TRUE, full.names = TRUE,
                         pattern = "\\.(hpp|h|hh|hxx|ipp|cpp|cc|dcf)$")
        if (!length(fs)) next
        fs <- sort(fs)
        info <- file.info(fs)
        parts <- c(parts, paste(sub(paste0("^", d, "/?"), "", fs),
                                info$size, as.numeric(info$mtime),
                                sep = "\x1f"))
    }
    if (!length(parts)) return("")
    # digest is not a dependency; a stable paste is enough, since it is only
    # ever compared for equality against the previous value in this session.
    paste(parts, collapse = "\x1e")
}

# -I flags for every contributed block (block dir for "<block>.hpp" + each vendored
# dependency dir). Folded into the compile flags by ai4bayescode_source(). Covers
# BOTH tiers: user-global installed blocks (~/.AI4BayesCode/blocks_download/) AND
# project-local blocks (./blocks_local/, relative to the working directory).
#' @keywords internal
#' @noRd
# The manifest fields skills/block_design_skills/skill.md Sec.3 requires of a
# contributed block. A bundle missing any of them, or missing its header, used
# to go on the compile -I path in silence -- an empty directory dropped into
# the library was reported as installed and was indistinguishable from a real
# block until the compile failed with a missing-header error.
#' @keywords internal
#' @noRd
.ai4b_required_manifest_fields <- c(
    "Block", "Version", "Title", "Description", "Author", "License",
    "EngineKind", "ConstraintKinds", "RoutingKey", "SelectWhen", "Skill",
    "Tests", "Depends")

#' Validate a contributed-block bundle
#'
#' @param path Bundle directory.
#' @return Character vector of problems; empty means the bundle is well formed.
#' @keywords internal
#' @noRd
.ai4b_validate_bundle <- function(path) {
    problems <- character(0)
    nm <- basename(normalizePath(path, winslash = "/", mustWork = FALSE))
    if (!dir.exists(path)) return(paste0(path, " is not a directory"))
    man_path <- file.path(path, "manifest.dcf")
    if (!file.exists(man_path)) {
        problems <- c(problems, "manifest.dcf is missing")
    } else {
        man <- tryCatch(read.dcf(man_path), error = function(e) NULL)
        if (is.null(man) || nrow(man) < 1L) {
            return(c(problems, "manifest.dcf could not be parsed"))
        }
        missing <- setdiff(.ai4b_required_manifest_fields, colnames(man))
        missing <- c(missing, Filter(function(f)
            f %in% colnames(man) && !nzchar(trimws(man[1, f])),
            .ai4b_required_manifest_fields))
        if (length(missing))
            problems <- c(problems, paste0(
                "manifest.dcf is missing required field(s): ",
                paste(unique(missing), collapse = ", ")))
        if ("Block" %in% colnames(man) && nzchar(man[1, "Block"]) &&
            man[1, "Block"] != nm)
            problems <- c(problems, sprintf(
                "manifest.dcf says Block: '%s' but the directory is '%s' -- they must match",
                man[1, "Block"], nm))
    }
    if (!file.exists(file.path(path, paste0(nm, ".hpp"))))
        problems <- c(problems, paste0(nm, ".hpp is missing (the header the block is used by)"))
    problems
}

.ai4b_block_cppflags <- function() {
    flags <- character(); seen <- character()
    # project-local FIRST, then user-global; DEDUP by block name so a locally
    # developed block shadows a same-named downloaded copy (the develop ->
    # upload -> re-download case) — local is your live working copy, the
    # download is the published snapshot, so local wins.
    for (bdir in c(file.path(getwd(), "blocks_local"), .ai4b_blocks_dir())) {
        if (!dir.exists(bdir)) next
        for (b in list.dirs(bdir, recursive = FALSE)) {
            nm <- basename(b)
            if (nm %in% seen) next
            seen <- c(seen, nm)
            probs <- .ai4b_validate_bundle(b)
            if (length(probs)) {
                warning("skipping block bundle '", b, "': ",
                        paste(probs, collapse = "; "), call. = FALSE)
                next
            }
            flags <- c(flags, paste0("-I", shQuote(b)))
            vdir <- file.path(b, "vendor")
            if (dir.exists(vdir))
                for (v in list.dirs(vdir, recursive = FALSE))
                    flags <- c(flags, paste0("-I", shQuote(v)))
        }
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
    # The GitHub API allows 60 unauthenticated requests per hour PER IP, which a
    # shared or institutional address can exhaust without the user doing
    # anything unusual. A token raises that to 5000/hr, and the failure it
    # prevents (HTTP 403) otherwise reads as an unexplained network error.
    hdrs <- c("User-Agent" = "AI4BayesCode-install_block")
    tok  <- Sys.getenv("GITHUB_PAT", Sys.getenv("GITHUB_TOKEN", ""))
    if (nzchar(tok)) hdrs <- c(hdrs, Authorization = paste("Bearer", tok))
    tr <- tryCatch(jsonlite::fromJSON(url(.ai4b_tree_url(), headers = hdrs)),
                   error = function(e) {
                       msg <- conditionMessage(e)
                       hint <- if (grepl("403|rate limit", msg, ignore.case = TRUE) &&
                                   !nzchar(tok))
                           paste0("\n  This looks like GitHub's unauthenticated ",
                                  "rate limit (60 requests/hour per IP). Set ",
                                  "GITHUB_PAT to a personal access token to raise it.")
                       else ""
                       stop("Could not reach the AI4BayesCode block registry: ",
                            msg, hint, call. = FALSE)
                   })
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

#' Contributed blocks visible to the compiler (like `installed.packages()`)
#'
#' @param tier Which tiers to list. BOTH reach the compile `-I` path:
#'   `"local"` is `./blocks_local/`, `"download"` is the per-user block
#'   library. A block present in both is listed once -- local shadows
#'   download, exactly as the compile path does. Default `"all"`.
#' @return Character vector of block names visible to the compiler.
#' @export
ai4bayescode_installed_blocks <- function(tier = c("all", "local", "download")) {
    tier <- match.arg(tier)
    dirs <- character(0)
    if (tier %in% c("all", "local"))    dirs <- c(dirs, file.path(getwd(), "blocks_local"))
    if (tier %in% c("all", "download")) dirs <- c(dirs, .ai4b_blocks_dir())
    out <- character(0)
    for (bdir in dirs) {
        if (!dir.exists(bdir)) next
        nms <- basename(list.dirs(bdir, recursive = FALSE))
        out <- c(out, setdiff(nms, out))     # local shadows download, as the -I path does
    }
    sort(out)
}

#' Install a contributed block from the hub registry (like `install.packages()`)
#'
#' Downloads a reviewed block bundle from the AI4BayesCode hub registry into the
#' per-user block library ([ai4bayescode_blocks_path()]). After install, the block
#' header and its vendored dependencies are on `ai4bayescode_source()`'s
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
    # Same validation remove_block gets: a raw stopifnot reported
    # `is.character(name) is not TRUE` for NULL, and NA slipped through to a
    # 404 with a raw URL in the warning.
    .ai4b_check_block_name(name)
    dest <- ai4bayescode_blocks_path(name)
    # A project-local block of the same name shadows the download (local > download):
    # the develop -> upload -> re-download case. Flag it so the user is not surprised
    # that edits to ./blocks_local/ still win over the freshly downloaded snapshot.
    if (!quiet && dir.exists(file.path(getwd(), "blocks_local", name)))
        message("Note: a project-local block '", name, "' exists in ./blocks_local/ and ",
                "will take PRECEDENCE over this download (local > download). The download ",
                "is kept as the published snapshot; remove the local copy to use it.")
    if (dir.exists(dest) && !force) {
        if (!quiet)
            message("Block '", name, "' is already installed (", dest, ").\n",
                    "  Use force = TRUE to reinstall, or ai4bayescode_remove_block(\"",
                    name, "\").")
        return(invisible(dest))
    }
    # 1. manifest (also the existence check)
    man_err <- NULL
    man_txt <- tryCatch(readLines(.ai4b_raw_url(sprintf("registry/%s/manifest.dcf", name)),
                                  warn = FALSE),
                        error = function(e) { man_err <<- conditionMessage(e); NULL })
    if (is.null(man_txt)) {
        # The fetch failing does NOT mean the block is absent -- being offline,
        # behind a proxy, or hitting a GitHub outage fails identically. The
        # registry INDEX settles it: if that is reachable, the block is really
        # not there; if it is not, this is a connectivity problem and saying
        # "not in the registry" sends the user hunting for a typo.
        idx_err <- NULL
        avail <- tryCatch(ai4bayescode_available_blocks(),
                          error = function(e) { idx_err <<- conditionMessage(e); NULL })
        if (is.null(avail)) {
            stop("Cannot reach the AI4BayesCode block registry, so whether '",
                 name, "' exists is unknown. Check your network connection ",
                 "(and any proxy), then retry.\n  manifest fetch: ", man_err,
                 "\n  registry index: ", idx_err, call. = FALSE)
        }
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
    .ai4b_check_block_name(name)   # explicit, so remove_block(NULL) reports the
                                   # name problem rather than the containment one
    dest <- ai4bayescode_blocks_path(name)
    # Defence in depth: whatever the name resolved to, it has to be a direct
    # child of the block library before anything is deleted recursively.
    # Compare the PARENT, not the block dir itself: normalizePath() only
    # resolves symlinks for a path that exists, so once the block is gone
    # (the remove-twice case) the block path stays "/var/..." while the
    # library root resolves to "/private/var/..." and an otherwise-correct
    # call looks like an escape. dirname() of the parent always exists.
    root   <- normalizePath(.ai4b_blocks_dir(), winslash = "/", mustWork = FALSE)
    parent <- normalizePath(dirname(dest),      winslash = "/", mustWork = FALSE)
    if (!identical(parent, root)) {
        stop("refusing to remove '", dest, "': not a block directory under ",
             root, call. = FALSE)
    }
    if (!dir.exists(dest)) {
        # A project-local block IS installed -- it is on the compile include
        # path and ai4bayescode_installed_blocks() lists it -- but it lives in
        # the user's own project, not the download cache, so this function does
        # not delete it. Saying "not installed" would flatly contradict what
        # installed_blocks() just reported.
        local_dir <- file.path("blocks_local", name)
        if (dir.exists(local_dir)) {
            message("Block '", name, "' is a project-local block at ",
                    normalizePath(local_dir, winslash = "/", mustWork = FALSE),
                    ".\n  ai4bayescode_remove_block() manages the download ",
                    "cache only; delete that directory to remove it.")
            return(invisible(FALSE))
        }
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
                "  (compile with ai4bayescode_source())")
    message("  -> usable now: the header + vendored deps are on the sourceCpp include path.")
    invisible()
}

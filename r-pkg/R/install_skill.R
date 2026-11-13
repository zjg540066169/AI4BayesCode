# install_skill.R -- install the AI4BayesCode agent skill for Claude Code or
# Codex from R, the same way on Windows, macOS and Linux. The skill is the
# whole GitHub repository; it is downloaded as a zip archive, so neither git
# nor any particular shell is needed.

.AI4B_SKILL_REPO <- "zjg540066169/AI4BayesCode"

# The home folder as the coding agents see it. On Windows R's "~" is the
# Documents folder (R_USER), while Claude Code and Codex use the profile
# folder (%USERPROFILE%).
#' @keywords internal
#' @noRd
.ai4b_home_dir <- function() {
    if (.Platform$OS.type == "windows") {
        up <- Sys.getenv("USERPROFILE")
        if (nzchar(up)) return(normalizePath(up, winslash = "/", mustWork = FALSE))
    }
    normalizePath(path.expand("~"), winslash = "/", mustWork = FALSE)
}

# Where each agent reads personal skills from.
#' @keywords internal
#' @noRd
.ai4b_skill_dest <- function(agent) {
    switch(agent,
        claude = {
            cfg  <- Sys.getenv("CLAUDE_CONFIG_DIR")
            base <- if (nzchar(cfg)) cfg else file.path(.ai4b_home_dir(), ".claude")
            file.path(base, "skills", "AI4BayesCode")
        },
        codex = file.path(.ai4b_home_dir(), ".agents", "skills", "AI4BayesCode"))
}

# TRUE when `d` holds an AI4BayesCode skill (or nothing), so replacing it
# cannot destroy anything else.
#' @keywords internal
#' @noRd
.ai4b_is_skill_dir <- function(d) {
    if (!length(list.files(d, all.files = TRUE, no.. = TRUE))) return(TRUE)
    f <- file.path(d, "SKILL.md")
    if (!file.exists(f)) return(FALSE)
    any(grepl("^name:\\s*AI4BayesCode\\s*$",
              readLines(f, n = 5L, warn = FALSE, encoding = "UTF-8")))
}

#' Install the AI4BayesCode skill for Claude Code or Codex
#'
#' Downloads the AI4BayesCode skill from GitHub and puts it where the coding
#' agent looks for personal skills: `~/.claude/skills/AI4BayesCode` for Claude
#' Code (`$CLAUDE_CONFIG_DIR/skills/AI4BayesCode` when that variable is set)
#' and `~/.agents/skills/AI4BayesCode` for Codex. On Windows `~` here is the
#' user's profile folder (`\%USERPROFILE\%`), which is where both agents look.
#' It works the same on Windows, macOS and Linux and needs no git. Run it again
#' to update the skill.
#'
#' An existing copy at the destination is replaced. A folder there that does
#' not hold the AI4BayesCode skill is never touched: the call stops instead.
#'
#' @param agent Which agent to install for: `"claude"`, `"codex"`, or both
#'   (the default).
#' @param ref Git branch or tag to install (default `"main"`).
#' @param quiet Suppress the progress messages (default `FALSE`).
#' @return Invisibly, the installed folder(s), named by agent.
#' @examples
#' \dontrun{
#' ai4bayescode_install_skill()           # Claude Code and Codex
#' ai4bayescode_install_skill("claude")   # Claude Code only
#' }
#' @export
ai4bayescode_install_skill <- function(agent = c("claude", "codex"), ref = "main",
                                       quiet = FALSE) {
    agent <- match.arg(agent, several.ok = TRUE)
    if (!is.character(ref) || length(ref) != 1L || is.na(ref) || !nzchar(ref))
        stop("`ref` must be a single branch or tag name, e.g. \"main\".", call. = FALSE)
    dests <- vapply(agent, .ai4b_skill_dest, character(1))
    for (d in dests) {
        if (dir.exists(d) && !.ai4b_is_skill_dir(d))
            stop("'", d, "' already exists and does not hold the AI4BayesCode ",
                 "skill. Move it away, then run this again.", call. = FALSE)
    }

    tmp <- tempfile("ai4b_skill_")
    dir.create(tmp)
    on.exit(unlink(tmp, recursive = TRUE, force = TRUE), add = TRUE)
    zip <- file.path(tmp, "skill.zip")
    url <- sprintf("https://github.com/%s/archive/%s.zip", .AI4B_SKILL_REPO, ref)
    if (!quiet) message("Downloading the AI4BayesCode skill (", ref, ") from GitHub ...")
    old <- options(timeout = max(600, getOption("timeout")))
    on.exit(options(old), add = TRUE)
    got <- tryCatch(utils::download.file(url, zip, mode = "wb", quiet = TRUE),
                    error = function(e) conditionMessage(e),
                    warning = function(w) conditionMessage(w))
    if (!(is.numeric(got) && got == 0) || !file.exists(zip))
        stop("Could not download ", url, "\n  ", got,
             "\n  Check the network connection (and any proxy), then retry.",
             call. = FALSE)
    utils::unzip(zip, exdir = tmp)
    top <- setdiff(list.dirs(tmp, recursive = FALSE), tmp)
    if (length(top) != 1L || !file.exists(file.path(top, "SKILL.md")))
        stop("The download from ", url, " does not contain the skill.", call. = FALSE)
    entries <- list.files(top, all.files = TRUE, no.. = TRUE, full.names = TRUE)

    for (d in dests) {
        if (dir.exists(d)) {
            # force: git marks pack files read-only, and Windows will not
            # delete a read-only file otherwise.
            unlink(d, recursive = TRUE, force = TRUE)
            if (dir.exists(d))
                stop("Could not remove the old copy at '", d, "'. Close any ",
                     "program that uses it, then run this again.", call. = FALSE)
        }
        dir.create(d, recursive = TRUE)
        ok <- file.copy(entries, d, recursive = TRUE, copy.date = TRUE)
        if (!all(ok) || !file.exists(file.path(d, "SKILL.md")))
            stop("Copying the skill to '", d, "' failed.", call. = FALSE)
    }

    if (!quiet) {
        message("Installed the AI4BayesCode skill:")
        if ("claude" %in% agent)
            message("  Claude Code: ", dests[["claude"]],
                    "\n    restart Claude Code, then type /AI4BayesCode")
        if ("codex" %in% agent) {
            message("  Codex:       ", dests[["codex"]],
                    "\n    restart Codex, then type $AI4BayesCode or pick it from /skills")
            legacy <- file.path(Sys.getenv("CODEX_HOME",
                                           unset = file.path(.ai4b_home_dir(), ".codex")),
                                "skills", "AI4BayesCode")
            if (dir.exists(legacy))
                message("  Note: an older copy is still in ", legacy, ". Codex reads ",
                        "that folder too and would list the skill twice, so delete it.")
        }
        message("Run ai4bayescode_install_skill() again to update.")
    }
    invisible(dests)
}

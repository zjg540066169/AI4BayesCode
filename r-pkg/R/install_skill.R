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

# The folder at `d`, through a symbolic link: an update replaces the folder
# a link points to and leaves the link itself alone.
#' @keywords internal
#' @noRd
.ai4b_real_dest <- function(d) {
    if (file.exists(d)) normalizePath(d, winslash = "/", mustWork = TRUE) else d
}

# One attempt at putting a copy of `entries` at `d`, with the staging folders
# in `work`. FALSE when it failed before anything at `d` changed, so that
# another `work` can be tried.
#' @keywords internal
#' @noRd
.ai4b_swap_in <- function(entries, d, work) {
    tag <- paste0(Sys.getpid(), "-", format(as.numeric(Sys.time()) * 1e6, scientific = FALSE))
    new <- file.path(work, paste0(".AI4BayesCode-new-", tag))
    old <- file.path(work, paste0(".AI4BayesCode-old-", tag))
    if (!dir.create(new, showWarnings = FALSE)) return(FALSE)
    ok <- all(file.copy(entries, new, recursive = TRUE, copy.date = TRUE)) &&
        file.exists(file.path(new, "SKILL.md"))
    if (!ok || (dir.exists(d) && !isTRUE(suppressWarnings(file.rename(d, old))))) {
        unlink(new, recursive = TRUE, force = TRUE)
        return(FALSE)
    }
    if (!isTRUE(suppressWarnings(file.rename(new, d)))) {
        if (dir.exists(old)) file.rename(old, d)
        unlink(new, recursive = TRUE, force = TRUE)
        stop("Could not move the new copy of the skill to '", d, "'.", call. = FALSE)
    }
    if (dir.exists(old)) {
        # force: git marks pack files read-only, and Windows will not delete a
        # read-only file otherwise.
        unlink(old, recursive = TRUE, force = TRUE)
        if (dir.exists(old))
            message("  Note: the old copy could not be deleted. Delete ", old, " by hand.")
    }
    TRUE
}

# Put a copy of `entries` (the files of the downloaded skill) at `d` without
# ever leaving `d` half deleted. The copy is made in a staging folder, the old
# folder is moved aside in one step (on Windows that fails as a whole while a
# file in it is open), the copy is moved in, and only then is the old folder
# deleted. The staging folders go beside the skills folder, out of the folder
# the agent lists skills from; when that place cannot be written or is another
# volume (a skills folder mounted on its own), they go inside the skills
# folder.
#' @keywords internal
#' @noRd
.ai4b_replace_with_copy <- function(entries, d) {
    dir.create(dirname(d), recursive = TRUE, showWarnings = FALSE)
    skills <- normalizePath(dirname(d), winslash = "/", mustWork = TRUE)
    for (work in unique(c(dirname(skills), skills)))
        if (.ai4b_swap_in(entries, d, work)) return(invisible(d))
    stop("Could not replace '", d, "'. If a program uses it (an editor, a terminal ",
         "whose current folder is inside it), close it, then run this again. The ",
         "installed skill is unchanged.", call. = FALSE)
}

# Download the skill at `ref` from GitHub and unpack it into `tmp`. Returns
# the unpacked folder that holds SKILL.md.
#' @keywords internal
#' @noRd
.ai4b_fetch_skill <- function(ref, tmp, quiet) {
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
    top
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
#' So does a git checkout of the skill (update that with git), unless
#' `force = TRUE`.
#'
#' @param agent Which agent to install for: `"claude"`, `"codex"`, or both
#'   (the default).
#' @param ref Git branch or tag to install (default `"main"`).
#' @param quiet Suppress the progress messages (default `FALSE`).
#' @param force Also replace a git checkout of the skill, losing any changes
#'   made in it (default `FALSE`).
#' @return Invisibly, the installed folder(s), named by agent.
#' @examples
#' \dontrun{
#' ai4bayescode_install_skill()           # Claude Code and Codex
#' ai4bayescode_install_skill("claude")   # Claude Code only
#' }
#' @export
ai4bayescode_install_skill <- function(agent = c("claude", "codex"), ref = "main",
                                       quiet = FALSE, force = FALSE) {
    agent <- match.arg(agent, several.ok = TRUE)
    if (!is.character(ref) || length(ref) != 1L || is.na(ref) || !nzchar(ref))
        stop("`ref` must be a single branch or tag name, e.g. \"main\".", call. = FALSE)
    dests <- vapply(agent, .ai4b_skill_dest, character(1))
    for (d in dests) {
        real <- .ai4b_real_dest(d)
        if (file.exists(real) && !(dir.exists(real) && .ai4b_is_skill_dir(real)))
            stop("'", d, "' already exists and does not hold the AI4BayesCode ",
                 "skill. Move it away, then run this again.", call. = FALSE)
        if (file.exists(file.path(real, ".git")) && !isTRUE(force))
            stop("'", real, "' is a git checkout of the skill. Update it with git:\n",
                 "  git -C \"", real, "\" pull --ff-only https://github.com/",
                 .AI4B_SKILL_REPO, ".git ", ref, "\n",
                 "or pass force = TRUE to replace it with a fresh copy, losing any ",
                 "changes made in it.", call. = FALSE)
    }

    tmp <- tempfile("ai4b_skill_")
    dir.create(tmp)
    on.exit(unlink(tmp, recursive = TRUE, force = TRUE), add = TRUE)
    top <- .ai4b_fetch_skill(ref, tmp, quiet)
    entries <- list.files(top, all.files = TRUE, no.. = TRUE, full.names = TRUE)

    for (d in dests) .ai4b_replace_with_copy(entries, .ai4b_real_dest(d))

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

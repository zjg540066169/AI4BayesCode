# ai4bayescode_install_skill(): what it may replace, and how. The download
# itself is not exercised here; .ai4b_replace_with_copy() is the step that
# swaps the files in.

old_skill <- function(d) {
    dir.create(d, recursive = TRUE)
    writeLines(c("---", "name: AI4BayesCode", "---"), file.path(d, "SKILL.md"))
    writeLines("old", file.path(d, "VERSION.txt"))
}

new_files <- function() {
    src <- tempfile("ai4b_new_")
    dir.create(src)
    writeLines(c("---", "name: AI4BayesCode", "---"), file.path(src, "SKILL.md"))
    writeLines("new", file.path(src, "VERSION.txt"))
    list.files(src, all.files = TRUE, no.. = TRUE, full.names = TRUE)
}

leftovers <- function(d) {
    # Staging folders left beside or inside the skills folder.
    c(list.files(dirname(dirname(d)), pattern = "^\\.AI4BayesCode-", all.files = TRUE),
      list.files(dirname(d), pattern = "^\\.AI4BayesCode-", all.files = TRUE))
}

test_that("a git checkout of the skill is left alone unless force = TRUE", {
    cfg <- tempfile("ai4b_cfg_")
    withr::local_envvar(CLAUDE_CONFIG_DIR = cfg)
    d <- file.path(cfg, "skills", "AI4BayesCode")
    old_skill(d)
    dir.create(file.path(d, ".git"))
    # Refused before anything is downloaded.
    expect_error(ai4bayescode_install_skill("claude", quiet = TRUE), "git checkout")
    expect_true(dir.exists(file.path(d, ".git")))
    expect_identical(readLines(file.path(d, "VERSION.txt")), "old")
})

test_that("an old copy is swapped for the new one, with nothing left over", {
    d <- file.path(tempfile("ai4b_home_"), ".claude", "skills", "AI4BayesCode")
    old_skill(d)
    AI4BayesCode:::.ai4b_replace_with_copy(new_files(), d)
    expect_identical(readLines(file.path(d, "VERSION.txt")), "new")
    expect_length(leftovers(d), 0L)
})

test_that("the old copy stays when it cannot be moved aside", {
    skip_on_os("windows")   # a read-only folder does not block a rename there
    skip_if(Sys.info()[["effective_user"]] == "root", "root ignores permissions")
    d <- file.path(tempfile("ai4b_home_"), ".claude", "skills", "AI4BayesCode")
    old_skill(d)
    Sys.chmod(dirname(d), "0555")
    on.exit(Sys.chmod(dirname(d), "0755"), add = TRUE)
    expect_error(AI4BayesCode:::.ai4b_replace_with_copy(new_files(), d), "unchanged")
    expect_identical(readLines(file.path(d, "VERSION.txt")), "old")
    expect_length(leftovers(d), 0L)
})

test_that("through a link, the folder it points to is updated and the link kept", {
    skip_on_os("windows")   # symbolic links need extra rights there
    home <- tempfile("ai4b_home_")
    target <- file.path(home, "dev copy", "AI4BayesCode")
    old_skill(target)
    d <- file.path(home, ".claude", "skills", "AI4BayesCode")
    dir.create(dirname(d), recursive = TRUE)
    file.symlink(target, d)
    AI4BayesCode:::.ai4b_replace_with_copy(new_files(), AI4BayesCode:::.ai4b_real_dest(d))
    expect_true(nzchar(Sys.readlink(d)))
    expect_identical(readLines(file.path(d, "VERSION.txt")), "new")
    expect_identical(readLines(file.path(target, "VERSION.txt")), "new")
})

fake_fetch <- function(ref, tmp, quiet) {
    top <- file.path(tmp, "AI4BayesCode-main")
    dir.create(top)
    writeLines(c("---", "name: AI4BayesCode", "---"), file.path(top, "SKILL.md"))
    writeLines("new", file.path(top, "VERSION.txt"))
    top
}

test_that("force = TRUE replaces a git checkout of the skill", {
    local_mocked_bindings(.ai4b_fetch_skill = fake_fetch, .package = "AI4BayesCode")
    cfg <- tempfile("ai4b_cfg_")
    withr::local_envvar(CLAUDE_CONFIG_DIR = cfg)
    d <- file.path(cfg, "skills", "AI4BayesCode")
    old_skill(d)
    dir.create(file.path(d, ".git"))
    expect_error(ai4bayescode_install_skill("claude", quiet = TRUE), "git checkout")
    ai4bayescode_install_skill("claude", quiet = TRUE, force = TRUE)
    expect_false(dir.exists(file.path(d, ".git")))
    expect_identical(readLines(file.path(d, "VERSION.txt")), "new")
    expect_length(leftovers(d), 0L)
})

test_that("the staging folders go inside the skills folder when beside it fails", {
    skip_on_os("windows")   # a read-only folder does not stop folder creation there
    skip_if(Sys.info()[["effective_user"]] == "root", "root ignores permissions")
    local_mocked_bindings(.ai4b_fetch_skill = fake_fetch, .package = "AI4BayesCode")
    cfg <- tempfile("ai4b_cfg_")
    withr::local_envvar(CLAUDE_CONFIG_DIR = cfg)
    d <- file.path(cfg, "skills", "AI4BayesCode")
    old_skill(d)
    Sys.chmod(cfg, "0555")
    on.exit(Sys.chmod(cfg, "0755"), add = TRUE)
    ai4bayescode_install_skill("claude", quiet = TRUE)
    expect_identical(readLines(file.path(d, "VERSION.txt")), "new")
    expect_length(list.files(dirname(d), pattern = "^\\.AI4BayesCode-", all.files = TRUE), 0L)
})

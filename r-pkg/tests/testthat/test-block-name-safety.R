# `ai4bayescode_remove_block()` must never delete anything but one block dir.
#
# `ai4bayescode_blocks_path(name)` is `file.path(library_root, name)`, and
# `file.path(root, "")` returns the library ROOT with a trailing separator. That
# passes `dir.exists()`, so `remove_block("")` used to `unlink(recursive=TRUE)`
# every installed block. `"../x"` escapes the library by the ordinary route.

local_block_library <- function(env = parent.frame()) {
    root <- file.path(tempfile("ai4b_blocks_"), "data_home")
    for (b in c("blockA", "blockB")) {
        dir.create(file.path(root, "blocks_download", b), recursive = TRUE)
        writeLines("// x", file.path(root, "blocks_download", b,
                                     paste0(b, ".hpp")))
    }
    dir.create(file.path(root, "PRECIOUS_USER_DATA"), recursive = TRUE)
    writeLines("do not delete",
               file.path(root, "PRECIOUS_USER_DATA", "keepme.txt"))
    withr::local_envvar(c(AI4BAYESCODE_DATA_HOME = root), .local_envir = env)
    root
}

# Each of these resolved to a real directory outside -- or above -- the single
# block it names.
bad_names <- function(root) {
    list("", ".", "..", "../PRECIOUS_USER_DATA", "a/b", "a\\b",
         NULL, NA_character_, 42, c("blockA", "blockB"),
         file.path(root, "PRECIOUS_USER_DATA"))
}

test_that("remove_block rejects every name that is not a plain block name", {
    root <- local_block_library()
    for (nm in bad_names(root)) {
        expect_error(ai4bayescode_remove_block(nm),
                     regexp = "block name must be",
                     info = paste("name:", paste(deparse(nm), collapse = "")))
    }
    expect_setequal(ai4bayescode_installed_blocks(), c("blockA", "blockB"))
    expect_true(file.exists(file.path(root, "PRECIOUS_USER_DATA", "keepme.txt")))
})

test_that("blocks_path rejects the same names, so install_block inherits it", {
    root <- local_block_library()
    for (nm in bad_names(root)) {
        if (is.null(nm)) next    # NULL is the documented "library root" form
        expect_error(ai4bayescode_blocks_path(nm), regexp = "block name must be")
    }
    # and the no-argument form still returns the library root
    expect_equal(ai4bayescode_blocks_path(),
                 file.path(root, "blocks_download"))
})

test_that("a real block still removes, and removing it twice is not an error", {
    local_block_library()
    expect_true(suppressMessages(ai4bayescode_remove_block("blockA")))
    expect_setequal(ai4bayescode_installed_blocks(), "blockB")
    expect_false(suppressMessages(ai4bayescode_remove_block("blockA")))
})

# ---------------------------------------------------------------------------
# A registry that cannot be reached is not the same as a block that is absent
# ---------------------------------------------------------------------------
# ai4bayescode_install_block() used to answer "Block 'x' is not in the registry"
# for ANY failure of the manifest fetch. Offline, behind a proxy, or during a
# GitHub outage the message is identical to a typo, so the user goes looking for
# a misspelling that is not there. The registry INDEX settles which it is.
local_dead_registry <- function(env = parent.frame()) {
    ns   <- asNamespace("AI4BayesCode")
    dead <- "https://127.0.0.1:9/unreachable"
    for (nm in c(".ai4b_raw_url", ".ai4b_tree_url")) {
        old <- get(nm, envir = ns)
        unlockBinding(nm, ns)
        assign(nm, function(...) dead, envir = ns)
        withr::defer(assign(nm, old, envir = ns), envir = env)
    }
}

test_that("an unreachable registry is not reported as a missing block", {
    local_dead_registry()
    err <- tryCatch(ai4bayescode_install_block("beta_gibbs_block"),
                    error = function(e) conditionMessage(e))
    expect_match(err, "Cannot reach")
    expect_match(err, "is unknown")
    expect_false(grepl("not in the registry", err))
})

test_that("a reachable registry still reports a genuinely absent block", {
    # The other direction: with the index reachable, an absent block must still
    # say so -- otherwise the fix would hide real typos behind a network error.
    ns  <- asNamespace("AI4BayesCode")
    old_raw <- get(".ai4b_raw_url", envir = ns)
    old_av  <- get("ai4bayescode_available_blocks", envir = ns)
    unlockBinding(".ai4b_raw_url", ns)
    unlockBinding("ai4bayescode_available_blocks", ns)
    assign(".ai4b_raw_url", function(...) "https://127.0.0.1:9/unreachable", envir = ns)
    assign("ai4bayescode_available_blocks", function(...) "beta_gibbs_block", envir = ns)
    withr::defer({
        assign(".ai4b_raw_url", old_raw, envir = ns)
        assign("ai4bayescode_available_blocks", old_av, envir = ns)
    })
    err <- tryCatch(ai4bayescode_install_block("no_such_block_xyz"),
                    error = function(e) conditionMessage(e))
    expect_match(err, "not in the registry")
    expect_match(err, "beta_gibbs_block")
})

# ---------------------------------------------------------------------------
# A project-local block is installed; it is just not the download cache's
# ---------------------------------------------------------------------------
test_that("removing a project-local block says where it is, not 'not installed'", {
    # ai4bayescode_installed_blocks() lists both tiers and the local tier reaches
    # the compiler, so answering "not installed" here flatly contradicts what the
    # user was just shown, and leaves them with no idea how to get rid of it.
    root <- local_block_library()
    proj <- withr::local_tempdir()
    dir.create(file.path(proj, "blocks_local", "toy_block"), recursive = TRUE)
    writeLines("// x", file.path(proj, "blocks_local", "toy_block", "toy_block.hpp"))
    withr::local_dir(proj)

    expect_true("toy_block" %in% ai4bayescode_installed_blocks())
    expect_message(res <- ai4bayescode_remove_block("toy_block"),
                   "project-local block")
    expect_false(res)
    expect_true(dir.exists(file.path(proj, "blocks_local", "toy_block")))
    # ... and the download cache is untouched
    expect_setequal(ai4bayescode_installed_blocks("download"), c("blockA", "blockB"))
})

test_that("a name in neither tier still reports 'not installed'", {
    local_block_library()
    expect_message(res <- ai4bayescode_remove_block("really_not_here"),
                   "is not installed")
    expect_false(res)
})

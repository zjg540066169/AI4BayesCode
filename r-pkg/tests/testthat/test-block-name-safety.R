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

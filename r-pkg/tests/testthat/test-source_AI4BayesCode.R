# Self-contained compile tests for ai4bayescode_source(). These are slow
# (each sourceCpp compiles a real example) and depend on a C++ toolchain, so
# they skip on CRAN.

test_that("ai4bayescode_source compiles an example from a file path", {
    skip_on_cran()
    skip_if_not_installed("Rcpp")
    skip_if_not_installed("RcppArmadillo")

    cpp <- ai4bayescode_examples_path("GaussianLocationScale.cpp")
    skip_if(!nzchar(cpp), "bundled example not found")

    ai4bayescode_source(cpp, rebuild = TRUE, env = environment())
    expect_true(exists("GaussianLocationScale"))

    set.seed(1)
    y <- rnorm(50, mean = 2, sd = 1)
    m <- new(GaussianLocationScale, y, 1L, FALSE)
    expect_silent(m$step(20L))
    cur <- m$get_current()
    expect_true(all(vapply(cur, function(x) all(is.finite(x)), logical(1))))
})

test_that("ai4bayescode_source accepts a source string", {
    skip_on_cran()
    skip_if_not_installed("Rcpp")
    skip_if_not_installed("RcppArmadillo")

    cpp <- ai4bayescode_examples_path("GaussianLocationScale.cpp")
    skip_if(!nzchar(cpp), "bundled example not found")
    src <- paste(readLines(cpp, warn = FALSE), collapse = "\n")

    # Compile from the string (no on-disk .cpp path passed).
    ai4bayescode_source(src, rebuild = TRUE, env = environment())
    expect_true(exists("GaussianLocationScale"))
})

test_that("ai4bayescode_source compiles examples needing eigen / celerite / libgp roots", {
    skip_on_cran()
    skip_if_not_installed("Rcpp")
    skip_if_not_installed("RcppArmadillo")

    # These three exercise the include roots the legacy plugin path omitted.
    for (ex in c("GMRFPrior",     # eigen
                 "GPTimeSeries",  # celerite
                 "GPRegression")) {  # libgp
        cpp <- ai4bayescode_examples_path(paste0(ex, ".cpp"))
        skip_if(!nzchar(cpp), paste0("bundled example not found: ", ex))
        e <- new.env()
        expect_error(
            ai4bayescode_source(cpp, rebuild = TRUE, env = e),
            NA,
            info = ex
        )
        expect_true(exists(ex, envir = e), info = ex)
    }
})

test_that("ai4bayescode_source errors clearly on a non-existent path that is not source", {
    expect_error(
        ai4bayescode_source("definitely_not_a_real_file.cpp"),
        regexp = "neither an existing file nor recognizable C\\+\\+ source"
    )
})

test_that("a file with a space in its name follows edits to its local headers", {
    skip_on_cran()
    skip_if_not_installed("Rcpp")
    skip_if_not_installed("RcppArmadillo")

    # Rcpp's dependency scan resolves #include "..." beside the file it builds,
    # so the space-free copy must sit beside the original.
    dir <- tempfile("ai4b_spaced_")
    dir.create(dir)
    hdr <- file.path(dir, "helper.hpp")
    writeLines("inline int helper_value() { return 1; }", hdr)
    cpp <- file.path(dir, "my model.cpp")
    writeLines(c("#include <Rcpp.h>", "#include \"helper.hpp\"",
                 "// [[Rcpp::export]]", "int spaced_value() { return helper_value(); }"),
               cpp)
    e <- new.env()
    ai4bayescode_source(cpp, env = e, quiet = TRUE)
    expect_identical(e$spaced_value(), 1L)
    Sys.sleep(1.1)   # a new mtime on a filesystem with one-second resolution
    writeLines("inline int helper_value() { return 2; }", hdr)
    ai4bayescode_source(cpp, env = e, quiet = TRUE)
    expect_identical(e$spaced_value(), 2L)
})

test_that("the build temp folder must be plain ASCII for the Windows toolchain", {
    # R's own code page is UTF-8 from R 4.2 on, but the assembler and linker
    # read paths in the system's ANSI code page, so only ASCII is always safe.
    ok <- AI4BayesCode:::.ai4b_ansi_ok
    expect_true(ok("C:/Users/runneradmin/tmp x/RtmpAB12"))
    expect_true(ok("C:\\Users\\RUNNER~1\\TMPX~1"))
    expect_false(ok(enc2utf8("C:/Users/runneradmin/tmp x \u7edf\u8ba1")))
    expect_false(ok(enc2utf8("C:/Users/Jos\u00e9/AppData/Local/Temp")))
})

test_that("processes staging the same spaced file at once never see a broken copy", {
    skip_on_cran()
    skip_on_os("windows")   # mclapply forks; the copy is renamed into place on every OS
    dir <- tempfile("ai4b_race_")
    dir.create(dir)
    cpp <- file.path(dir, "my model.cpp")
    writeLines(rep("// some generated sampler code, a reasonably long line of text", 3000), cpp)
    ref <- unname(tools::md5sum(cpp))
    staged <- file.path(dir, ".ai4b_my_model.cpp")
    for (round in 1:10) {
        unlink(staged)
        go <- as.numeric(Sys.time()) + 0.3   # all workers start at the same moment
        seen <- parallel::mclapply(1:2, function(i) {   # R CMD check allows two
            while (as.numeric(Sys.time()) < go) {}
            p <- AI4BayesCode:::.ai4b_stage_spaced_file(cpp)
            identical(unname(tools::md5sum(p)), ref)
        }, mc.cores = 2)
        expect_true(all(unlist(seen)), info = paste("round", round))
    }
    expect_length(list.files(dir, pattern = "^\\.ai4b_tmp_", all.files = TRUE), 0L)
})

test_that("the staged copy of a read-only file stays writable and follows edits", {
    dir <- tempfile("ai4b_ro_")
    dir.create(dir)
    cpp <- file.path(dir, "my model.cpp")
    writeLines("// one", cpp)
    Sys.chmod(cpp, "0444")
    p1 <- AI4BayesCode:::.ai4b_stage_spaced_file(cpp)
    expect_identical(dirname(p1), normalizePath(dir, winslash = "/"))
    Sys.chmod(cpp, "0644")
    writeLines("// two", cpp)
    p2 <- AI4BayesCode:::.ai4b_stage_spaced_file(cpp)
    expect_identical(p2, p1)
    expect_identical(readLines(p2), "// two")
})

# A module whose function returns the address modulo 64 of Armadillo heap
# arrays of 17 to 400 elements. `depends` is the list of packages in its
# // [[Rcpp::depends(...)]] line.
arma_probe_file <- function(depends = "RcppArmadillo") {
    f <- tempfile("ai4b_align_", fileext = ".cpp")
    writeLines(c(
        paste0("// [[Rcpp::depends(", depends, ")]]"),
        "#include <RcppArmadillo.h>",
        "#include <cstdint>",
        "#include <vector>",
        "// [[Rcpp::export]]",
        "Rcpp::IntegerVector ai4b_arma_remainders() {",
        "    std::vector<int> out;",
        "    std::vector<arma::vec> keep;",
        "    for (arma::uword n = 17; n <= 400; ++n) {",
        "        keep.emplace_back(n, arma::fill::zeros);",
        "        arma::mat A(n, 2, arma::fill::ones);",
        "        arma::vec v = A.col(0) + 1.0;",
        "        out.push_back(int(reinterpret_cast<std::uintptr_t>(keep.back().memptr()) % 64));",
        "        out.push_back(int(reinterpret_cast<std::uintptr_t>(A.memptr()) % 64));",
        "        out.push_back(int(reinterpret_cast<std::uintptr_t>(v.memptr()) % 64));",
        "    }",
        "    return Rcpp::wrap(out);",
        "}"), f)
    f
}

test_that("the build fingerprint follows the flags and the force-included header", {
    fp  <- AI4BayesCode:::.ai4b_build_fingerprint
    hdr <- tempfile(fileext = ".hpp")
    writeLines("// one", hdr)
    a <- fp(c("-Iinc", "-DX"), "-lm", hdr)
    expect_match(a, "^[0-9a-f]{12}$")
    expect_identical(fp(c("-Iinc", "-DX"), "-lm", hdr), a)
    expect_false(identical(
        fp(c("-Iinc", "-DX", "-include AI4BayesCode/arma_aligned_alloc.hpp"), "-lm", hdr), a))
    expect_false(identical(fp(c("-Iinc", "-DX"), "-lblas", hdr), a))
    writeLines("// two", hdr)
    expect_false(identical(fp(c("-Iinc", "-DX"), "-lm", hdr), a))
})

test_that("Armadillo heap arrays of a compiled model lie on 64-byte boundaries", {
    skip_on_cran()
    skip_if_not_installed("Rcpp")
    skip_if_not_installed("RcppArmadillo")
    e <- new.env()
    ai4bayescode_source(arma_probe_file(), env = e, quiet = TRUE)
    rem <- e$ai4b_arma_remainders()
    expect_length(rem, 3L * 384L)
    expect_true(all(rem == 0L))
})

test_that("a module built earlier in the same cacheDir with other flags is not reused", {
    skip_on_cran()
    skip_if_not_installed("Rcpp")
    skip_if_not_installed("RcppArmadillo")
    cache <- tempfile("ai4b_cache_")
    dir.create(cache)
    f <- arma_probe_file()
    # Rcpp::sourceCpp on its own: no -include, Armadillo's own allocator.
    Rcpp::sourceCpp(f, cacheDir = cache, env = new.env())
    e <- new.env()
    ai4bayescode_source(f, env = e, quiet = TRUE, cacheDir = cache)
    expect_true(all(e$ai4b_arma_remainders() == 0L))
    expect_length(list.files(cache, pattern = "^ai4b-[0-9a-f]{12}$"), 1L)
})

test_that("the Rcpp plugin force-includes the alignment header", {
    flags <- inlineCxxPlugin()$env$PKG_CPPFLAGS
    expect_match(flags, "-include AI4BayesCode/arma_aligned_alloc.hpp", fixed = TRUE)
    expect_true(file.exists(system.file("include", "AI4BayesCode", "arma_aligned_alloc.hpp",
                                        package = "AI4BayesCode")))
})

test_that("a file compiled through the Rcpp plugin has 64-byte-aligned Armadillo heap arrays", {
    skip_on_cran()
    skip_if_not_installed("Rcpp")
    skip_if_not_installed("RcppArmadillo")
    # // [[Rcpp::depends(AI4BayesCode)]] makes Rcpp::sourceCpp() call
    # AI4BayesCode::inlineCxxPlugin() for the compiler flags.
    f <- arma_probe_file("RcppArmadillo, AI4BayesCode")
    e <- new.env()
    Rcpp::sourceCpp(f, env = e, cacheDir = tempfile("ai4b_plugin_cache_"))
    rem <- e$ai4b_arma_remainders()
    expect_length(rem, 3L * 384L)
    expect_true(all(rem == 0L))
})

test_that("a failed build reports the call Rcpp::sourceCpp(build_file, ...)", {
    skip_on_cran()
    skip_if_not_installed("Rcpp")
    skip_if_not_installed("RcppArmadillo")
    f <- tempfile("ai4b_broken_", fileext = ".cpp")
    writeLines(c("#include <Rcpp.h>",
                 "// [[Rcpp::export]]",
                 "int ai4b_broken() { return ai4b_undeclared_name; }"), f)
    err <- NULL
    utils::capture.output(err <- tryCatch(
        suppressMessages(suppressWarnings(
            ai4bayescode_source(f, env = new.env(), quiet = TRUE))),
        error = identity))
    expect_s3_class(err, "error")
    expect_identical(deparse(conditionCall(err)[[1L]]), "Rcpp::sourceCpp")
    expect_identical(conditionCall(err)[[2L]], as.name("build_file"))
})

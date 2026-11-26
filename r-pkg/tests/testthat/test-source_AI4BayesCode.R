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

# A module with three functions. `depends` is the list of packages in its
# // [[Rcpp::depends(...)]] line.
#   ai4b_arma_alignment()  AI4BayesCode::arma_heap_alignment if the build
#                          force-included AI4BayesCode/arma_aligned_alloc.hpp
#                          (256 on macOS, 0 on other systems), -1 otherwise
#   ai4b_arma_addresses()  addresses modulo 65536 of Armadillo heap arrays of
#                          17 to 400 elements
#   ai4b_offset_results()  results of ddot (n = 5000, 14400), dtrtrs (lower,
#                          n = 110, one right-hand side) and dgetrf + dgetrs
#                          (n = 110, one right-hand side) with every operand at
#                          offset k * 64 bytes from a 16384-byte boundary,
#                          k = 0, ..., 255: column k + 1 of each matrix is the
#                          result at offset `offsets[k + 1]`
arma_probe_file <- function(depends = "RcppArmadillo") {
    f <- tempfile("ai4b_align_", fileext = ".cpp")
    writeLines(c(
        paste0("// [[Rcpp::depends(", depends, ")]]"),
        "#include <RcppArmadillo.h>",
        "#include <algorithm>",
        "#include <cmath>",
        "#include <cstdint>",
        "#include <cstring>",
        "#include <random>",
        "#include <vector>",
        "// [[Rcpp::export]]",
        "int ai4b_arma_alignment() {",
        "#ifdef AI4BAYESCODE_ARMA_ALIGNED_ALLOC_HPP",
        "    return static_cast<int>(AI4BayesCode::arma_heap_alignment);",
        "#else",
        "    return -1;",
        "#endif",
        "}",
        "// [[Rcpp::export]]",
        "Rcpp::IntegerVector ai4b_arma_addresses() {",
        "    std::vector<int> out;",
        "    std::vector<arma::vec> keep;",
        "    for (arma::uword n = 17; n <= 400; ++n) {",
        "        keep.emplace_back(n, arma::fill::zeros);",
        "        arma::mat A(n, 2, arma::fill::ones);",
        "        arma::vec v = A.col(0) + 1.0;",
        "        for (const double* p : {keep.back().memptr(), A.memptr(), v.memptr()})",
        "            out.push_back(int(reinterpret_cast<std::uintptr_t>(p) % 65536));",
        "    }",
        "    return Rcpp::wrap(out);",
        "}",
        "// room for n_bytes at every offset k * 64, k = 0..255, from a 16384-byte boundary",
        "struct OffsetBuffer {",
        "    std::vector<unsigned char> raw;",
        "    unsigned char* base;",
        "    explicit OffsetBuffer(std::size_t n_bytes) : raw(n_bytes + 2 * 16384) {",
        "        const std::uintptr_t p = reinterpret_cast<std::uintptr_t>(raw.data());",
        "        base = raw.data() + (16384 - p % 16384) % 16384;",
        "    }",
        "    double* at(std::size_t off) { return reinterpret_cast<double*>(base + off); }",
        "};",
        "static std::vector<double> draws(std::size_t n, unsigned seed) {",
        "    std::mt19937_64 rng(seed);",
        "    std::normal_distribution<double> nd;",
        "    std::vector<double> v(n);",
        "    for (auto& x : v) x = nd(rng);",
        "    return v;",
        "}",
        "// [[Rcpp::export]]",
        "Rcpp::List ai4b_offset_results() {",
        "    const int K = 256;",
        "    Rcpp::IntegerVector offsets(K);",
        "    for (int k = 0; k < K; ++k) offsets[k] = 64 * k;",
        "    Rcpp::NumericMatrix d1(1, K), d2(1, K);",
        "    for (int j = 0; j < 2; ++j) {",
        "        const std::size_t n = (j == 0) ? 5000 : 14400;",
        "        const std::vector<double> x0 = draws(n, 21), y0 = draws(n, 22);",
        "        OffsetBuffer bx(n * 8), by(n * 8);",
        "        for (int k = 0; k < K; ++k) {",
        "            double* x = bx.at(64 * k); double* y = by.at(64 * k);",
        "            std::memcpy(x, x0.data(), n * 8); std::memcpy(y, y0.data(), n * 8);",
        "            (j == 0 ? d1 : d2)(0, k) = arma::blas::dot(arma::uword(n), x, y);",
        "        }",
        "    }",
        "    // not const: Armadillo releases before 15.8 declare the LAPACK",
        "    // wrappers with non-const pointer parameters",
        "    arma::blas_int n = 110, n_rhs = 1;",
        "    const std::size_t nn = 110, n2 = nn * nn;",
        "    const std::vector<double> a0 = draws(n2, 23), b0 = draws(nn, 24);",
        "    std::vector<double> L0(n2, 0.0), G0(a0);",
        "    for (std::size_t j = 0; j < nn; ++j) {",
        "        for (std::size_t i = j; i < nn; ++i)",
        "            L0[i + j * nn] = (i == j) ? 5.0 + std::fabs(a0[i + j * nn]) : 0.3 * a0[i + j * nn];",
        "        G0[j + j * nn] += double(nn);",
        "    }",
        "    Rcpp::NumericMatrix tr(nn, K), ge(nn, K);",
        "    OffsetBuffer bL(n2 * 8), bG(n2 * 8), bb(nn * 8), bp(nn * sizeof(arma::blas_int));",
        "    for (int k = 0; k < K; ++k) {",
        "        const std::size_t off = 64 * k;",
        "        char uplo = 'L', trans = 'N', diag = 'N';",
        "        arma::blas_int info = 0;",
        "        double* L = bL.at(off); double* b = bb.at(off);",
        "        std::memcpy(L, L0.data(), n2 * 8); std::memcpy(b, b0.data(), nn * 8);",
        "        arma::lapack::trtrs<double>(&uplo, &trans, &diag, &n, &n_rhs, L, &n, b, &n, &info);",
        "        std::copy(b, b + nn, tr.column(k).begin());",
        "        double* G = bG.at(off);",
        "        arma::blas_int* ipiv = reinterpret_cast<arma::blas_int*>(bp.at(off));",
        "        std::memcpy(G, G0.data(), n2 * 8); std::memcpy(b, b0.data(), nn * 8);",
        "        arma::lapack::getrf<double>(&n, &n, G, &n, ipiv, &info);",
        "        arma::lapack::getrs<double>(&trans, &n, &n_rhs, G, &n, ipiv, b, &n, &info);",
        "        std::copy(b, b + nn, ge.column(k).begin());",
        "    }",
        "    return Rcpp::List::create(Rcpp::Named(\"offsets\") = offsets,",
        "        Rcpp::Named(\"ddot_5000\") = d1, Rcpp::Named(\"ddot_14400\") = d2,",
        "        Rcpp::Named(\"dtrtrs_110\") = tr, Rcpp::Named(\"dgetrs_110\") = ge);",
        "}"), f)
    f
}

# Every offset in one class of the offset modulo `alignment` gives the same
# bits. With an alignment below the period of a routine's dependence on the
# operand addresses (256 bytes for these routines and data on an Apple M2 Max
# with macOS 27.0.1 Accelerate), offsets of one class give different results.
expect_one_result_per_class <- function(res, alignment) {
    cls <- res$offsets %% alignment
    for (nm in setdiff(names(res), "offsets")) {
        M <- res[[nm]]
        for (cl in unique(cls)) {
            cols <- which(cls == cl)
            same <- vapply(cols, function(j) identical(M[, j], M[, cols[1L]], num.eq = FALSE),
                           logical(1))
            expect_true(all(same),
                        label = sprintf("%s: one result for the offsets %d modulo %d",
                                        nm, cl, alignment))
        }
    }
}

# The header forces the 256-byte alignment on macOS only; elsewhere it leaves
# Armadillo's allocation as it is and arma_heap_alignment is 0.
on_macos <- identical(Sys.info()[["sysname"]], "Darwin")
expected_alignment <- if (on_macos) 256L else 0L

test_that("the build fingerprint follows the flags, the force-included header and the package version", {
    fp  <- AI4BayesCode:::.ai4b_build_fingerprint
    hdr <- tempfile(fileext = ".hpp")
    writeLines("// one", hdr)
    a <- fp(c("-Iinc", "-DX"), "-lm", hdr)
    expect_match(a, "^[0-9a-f]{12}$")
    expect_identical(fp(c("-Iinc", "-DX"), "-lm", hdr), a)
    # the installed version is the default
    expect_identical(fp(c("-Iinc", "-DX"), "-lm", hdr,
                        version = as.character(utils::packageVersion("AI4BayesCode"))), a)
    expect_false(identical(fp(c("-Iinc", "-DX"), "-lm", hdr, version = "1.1.2"), a))
    expect_false(identical(
        fp(c("-Iinc", "-DX", "-include AI4BayesCode/arma_aligned_alloc.hpp"), "-lm", hdr), a))
    expect_false(identical(fp(c("-Iinc", "-DX"), "-lblas", hdr), a))
    writeLines("// two", hdr)
    expect_false(identical(fp(c("-Iinc", "-DX"), "-lm", hdr), a))
})

# The probe module built once by ai4bayescode_source() for the two tests below.
probe_env <- local({
    e <- NULL
    function() {
        if (is.null(e)) {
            e <<- new.env()
            ai4bayescode_source(arma_probe_file(), env = e, quiet = TRUE)
        }
        e
    }
})

test_that("Armadillo heap arrays of a compiled model lie on arma_heap_alignment-byte boundaries", {
    skip_on_cran()
    skip_if_not_installed("Rcpp")
    skip_if_not_installed("RcppArmadillo")
    e <- probe_env()
    alignment <- e$ai4b_arma_alignment()
    expect_identical(alignment, expected_alignment)
    skip_if(!on_macos, "the header sets the allocator on macOS only")
    addr <- e$ai4b_arma_addresses()
    expect_length(addr, 3L * 384L)
    expect_true(all(addr %% alignment == 0L))
})

test_that("ddot, dtrtrs and dgetrs give one result per class of the operand offset modulo arma_heap_alignment", {
    skip_on_cran()
    skip_if_not_installed("Rcpp")
    skip_if_not_installed("RcppArmadillo")
    skip_if(!on_macos, "the header sets the allocator on macOS only")
    e <- probe_env()
    expect_one_result_per_class(e$ai4b_offset_results(), e$ai4b_arma_alignment())
})

test_that("a module built earlier in the same cacheDir with other flags is not reused", {
    skip_on_cran()
    skip_if_not_installed("Rcpp")
    skip_if_not_installed("RcppArmadillo")
    cache <- tempfile("ai4b_cache_")
    dir.create(cache)
    f <- arma_probe_file()
    # Rcpp::sourceCpp on its own: no -include, Armadillo's own allocator.
    e0 <- new.env()
    Rcpp::sourceCpp(f, cacheDir = cache, env = e0)
    expect_identical(e0$ai4b_arma_alignment(), -1L)
    e <- new.env()
    ai4bayescode_source(f, env = e, quiet = TRUE, cacheDir = cache)
    alignment <- e$ai4b_arma_alignment()
    expect_identical(alignment, expected_alignment)
    if (on_macos) expect_true(all(e$ai4b_arma_addresses() %% alignment == 0L))
    expect_length(list.files(cache, pattern = "^ai4b-[0-9a-f]{12}$"), 1L)
})

test_that("the Rcpp plugin force-includes the alignment header", {
    flags <- inlineCxxPlugin()$env$PKG_CPPFLAGS
    expect_match(flags, "-include AI4BayesCode/arma_aligned_alloc.hpp", fixed = TRUE)
    expect_true(file.exists(system.file("include", "AI4BayesCode", "arma_aligned_alloc.hpp",
                                        package = "AI4BayesCode")))
})

test_that("the Rcpp plugin writes an include root with a space in double quotes", {
    # Unquoted, the compiler receives the parts of such a path as separate
    # arguments ("no such file or directory: 'with'" for a library folder
    # "lib with space" on macOS). A path without a space is written as before.
    f <- AI4BayesCode:::.ai4b_plugin_path
    expect_identical(f("/lib/AI4BayesCode/include", windows = FALSE),
                     "/lib/AI4BayesCode/include")
    expect_identical(f("/lib with space/AI4BayesCode/include", windows = FALSE),
                     "\"/lib with space/AI4BayesCode/include\"")
    expect_identical(f("C:\\R\\library\\AI4BayesCode\\include", windows = FALSE),
                     "C:/R/library/AI4BayesCode/include")
    flags <- inlineCxxPlugin()$env$PKG_CPPFLAGS
    inc <- system.file("include", package = "AI4BayesCode")
    expect_true(grepl(paste0("-I", f(inc)), flags, fixed = TRUE))
})

test_that("a file compiled through the Rcpp plugin has aligned Armadillo heap arrays and one routine result per offset class", {
    skip_on_cran()
    skip_if_not_installed("Rcpp")
    skip_if_not_installed("RcppArmadillo")
    # // [[Rcpp::depends(AI4BayesCode)]] makes Rcpp::sourceCpp() call
    # AI4BayesCode::inlineCxxPlugin() for the compiler flags. The module links
    # R's LAPACK and BLAS (RcppArmadillo's plugin), not -framework Accelerate.
    f <- arma_probe_file("RcppArmadillo, AI4BayesCode")
    e <- new.env()
    Rcpp::sourceCpp(f, env = e, cacheDir = tempfile("ai4b_plugin_cache_"))
    alignment <- e$ai4b_arma_alignment()
    expect_identical(alignment, expected_alignment)
    skip_if(!on_macos, "the header sets the allocator on macOS only")
    addr <- e$ai4b_arma_addresses()
    expect_length(addr, 3L * 384L)
    expect_true(all(addr %% alignment == 0L))
    expect_one_result_per_class(e$ai4b_offset_results(), alignment)
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

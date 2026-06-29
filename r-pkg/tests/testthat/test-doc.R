# ai4bayescode_doc() parses usage from .cpp source (no compilation needed).

test_that("ai4bayescode_doc parses a sampler class from a .cpp path", {
    cpp <- ai4bayescode_examples_path("GaussianLocationScale.cpp")
    skip_if(!nzchar(cpp), "bundled example not found")

    info <- ai4bayescode_doc(cpp)
    expect_equal(info$class, "GaussianLocationScale")
    nms <- vapply(info$constructor, function(a) a$name, "")
    expect_true(all(c("y", "rng_seed", "keep_history") %in% nms))
    # the optional arg carries its default
    kh <- Filter(function(a) a$name == "keep_history", info$constructor)[[1]]
    expect_false(is.na(kh$default))
})

test_that("ai4bayescode_doc handles a [[Rcpp::export]] function file", {
    cpp <- ai4bayescode_examples_path("SpikeSlabSinhBijection.cpp")
    skip_if(!nzchar(cpp), "bundled example not found")

    info <- ai4bayescode_doc(cpp)
    expect_true(is.na(info$class))
    expect_true("spike_slab_sinh_chain" %in% names(info$exports))
})

test_that("ai4bayescode_doc surfaces @example:R and falls back when absent", {
    f <- tempfile(fileext = ".cpp")
    writeLines(c(
        "// ============================================================",
        "//  Fix.cpp -- toy model",
        "//  y ~ Normal(mu, sigma).",
        "// @example:R",
        "//   library(AI4BayesCode)",
        "//   y <- rnorm(50)",
        "//   m <- new(Fix, y, 1L)",
        "//   m$step(100)",
        "// @example:python",
        "//   import numpy as np, AI4BayesCode",
        "//   m = mod.Fix(np.random.randn(50), 1)",
        "// @example:end",
        "// ============================================================",
        "// [[Rcpp::depends(RcppArmadillo)]]",
        "#include <RcppArmadillo.h>",
        "class Fix { public: Fix(const arma::vec& y, int rng_seed){} void step(int n){} };",
        "RCPP_MODULE(Fix_module){ Rcpp::class_<Fix>(\"Fix\").constructor<arma::vec,int>().method(\"step\", &Fix::step); }"
    ), f)
    info <- ai4bayescode_doc(f)
    # the R block is surfaced; the python block does NOT leak into the R card
    expect_true(any(grepl("rnorm", info$example)))
    expect_false(any(grepl("np.random", info$example)))
    # the @example block is NOT swallowed into the model description
    expect_false(any(grepl("@example|rnorm", info$description)))
    # internal parser also extracts the python block on demand
    src <- paste(readLines(f), collapse = "\n")
    expect_true(any(grepl("np.random", AI4BayesCode:::.ai4b_parse_example(src, "python"))))

    # a file with no @example block -> example is NULL (auto-stub fallback)
    g <- tempfile(fileext = ".cpp")
    writeLines(c(
        "//  G.cpp -- m",
        "// [[Rcpp::depends(RcppArmadillo)]]",
        "#include <RcppArmadillo.h>",
        "class G { public: G(const arma::vec& y){} void step(int n){} };",
        "RCPP_MODULE(G_module){ Rcpp::class_<G>(\"G\").constructor<arma::vec>().method(\"step\", &G::step); }"
    ), g)
    expect_null(ai4bayescode_doc(g)$example)
})

test_that("every bundled example renders a usable doc card", {
    files <- list.files(ai4bayescode_examples_path(), pattern = "\\.cpp$",
                        full.names = TRUE)
    skip_if(length(files) == 0L, "no bundled examples")
    bad <- character(0)
    for (f in files) {
        info <- tryCatch({ utils::capture.output(r <- ai4bayescode_doc(f)); r },
                         error = function(e) e)
        ok <- !inherits(info, "error") &&
            (length(info$constructor) > 0 || length(info$exports) > 0)
        if (!ok) bad <- c(bad, basename(f))
    }
    expect_equal(bad, character(0))
})

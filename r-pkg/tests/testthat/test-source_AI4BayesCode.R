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

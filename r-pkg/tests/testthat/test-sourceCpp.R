# Compile + load tests. These are slow (~30s each on first run because
# we sourceCpp a real example) — skip on CRAN.

test_that("ai4bayescode_example compiles GaussianLocationScale", {
    skip_on_cran()
    skip_if_not_installed("Rcpp")
    skip_if_not_installed("RcppArmadillo")

    # Should compile without error and register the Rcpp module.
    expect_silent({
        ai4bayescode_example("GaussianLocationScale")
    })

    # Class should now be available in the test environment.
    expect_true(exists("GaussianLocationScale"))
})

test_that("compiled GaussianLocationScale can step + return finite results", {
    skip_on_cran()

    ai4bayescode_example("GaussianLocationScale")

    set.seed(1)
    y <- rnorm(50, mean = 2, sd = 1.0)
    m <- new(GaussianLocationScale, y, seed = 1L, keep_history = FALSE)
    expect_silent(m$step(10L))

    cur <- m$get_current()
    expect_true(is.list(cur))
    expect_true(all(sapply(cur, function(x) all(is.finite(x)))))
})

test_that("ai4bayescode_example errors on unknown name", {
    expect_error(
        ai4bayescode_example("ThisExampleDoesNotExist"),
        regexp = "not found"
    )
})

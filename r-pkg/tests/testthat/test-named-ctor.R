# Named-argument constructors for Rcpp module classes.
#
# Rcpp dispatches a module constructor on ARITY ALONE and drops argument
# names, so these tests pin the layer that recovers them: the signature is
# parsed from the .cpp, arguments are matched the way base R matches them,
# and the chosen bound registration is identified by TYPE, not by arity.
# The matching layer is pure R, so none of this needs a compiler.

sig_of <- function(file, class_name) {
    cpp <- ai4bayescode_examples_path(file)
    skip_if(!nzchar(cpp), "bundled example not found")
    AI4BayesCode:::.ai4b_ctor_signatures(cpp, class_name)
}

test_that("C++ default literals convert to R values by declared type", {
    f <- AI4BayesCode:::.ai4b_cpp_default_to_r
    expect_identical(f("false", "bool")[[1]], FALSE)
    expect_identical(f("true", "bool")[[1]], TRUE)
    expect_identical(f("50", "int")[[1]], 50L)
    expect_identical(f("100", "std::size_t")[[1]], 100L)
    expect_identical(f("0.001", "double")[[1]], 0.001)
    expect_identical(f("1.0", "double")[[1]], 1.0)
    expect_identical(f("1e-6", "double")[[1]], 1e-6)
    expect_identical(f("3u", "int")[[1]], 3L)            # literal suffix
    # A default-constructed armadillo container is empty and reconstructible.
    expect_identical(f("arma::vec()", "const arma::vec&")[[1]], numeric(0))
    # Anything we cannot reconstruct leaves the argument REQUIRED rather than
    # inventing a value the user never wrote.
    expect_null(f("compute_default(y)", "double"))
    expect_null(f(NA_character_, "double"))
})

test_that("a constructor signature carries names, defaults and required flags", {
    sg <- sig_of("BartNoise.cpp", "BartNoise")
    expect_gte(length(sg), 1L)
    s <- sg[[1]]
    expect_identical(s$names[1:2], c("X", "y"))
    # data is required; every tuning knob carries a default
    expect_false(any(s$has_default[1:2]))
    expect_true(all(s$has_default[-(1:2)]))
    expect_identical(s$defaults[[which(s$names == "ntrees")]], 200L)
    expect_identical(s$defaults[[which(s$names == "base")]], 0.95)
    expect_identical(s$defaults[[which(s$names == "dart")]], FALSE)
})

test_that("argument matching follows base R's rules exactly", {
    nms  <- c("X", "y", "Z", "sigma_a", "pi_a", "pi_b", "rng_seed")
    defs <- list(NULL, NULL, NULL, 0.001, 1, 1, 1L)
    has  <- c(FALSE, FALSE, FALSE, TRUE, TRUE, TRUE, TRUE)
    mm   <- function(...) AI4BayesCode:::.ai4b_match_ctor_args(
        nms, defs, has, list(...), "T")$values
    ref  <- function(X, y, Z, sigma_a = 0.001, pi_a = 1, pi_b = 1, rng_seed = 1L)
        list(X, y, Z, sigma_a, pi_a, pi_b, rng_seed)

    expect_equal(mm(1, 2, 3),                      ref(1, 2, 3))
    expect_equal(mm(X = 1, y = 2, Z = 3),          ref(X = 1, y = 2, Z = 3))
    expect_equal(mm(rng_seed = 9L, Z = 3, y = 2, X = 1),
                 ref(rng_seed = 9L, Z = 3, y = 2, X = 1))
    # a named argument is bound first, positionals then fill what is left
    expect_equal(mm(1, 2, 3, rng_seed = 9L),       ref(1, 2, 3, rng_seed = 9L))
    expect_equal(mm(1, 2, 3, X = 9),               ref(1, 2, 3, X = 9))
    # unique partial matching, as in base R
    expect_equal(mm(1, 2, 3, sig = 0.5),           ref(1, 2, 3, sig = 0.5))
})

test_that("a call that base R would reject is rejected here too", {
    nms  <- c("X", "y", "pi_a", "pi_b")
    defs <- list(NULL, NULL, 1, 1)
    has  <- c(FALSE, FALSE, TRUE, TRUE)
    mm   <- function(...) AI4BayesCode:::.ai4b_match_ctor_args(
        nms, defs, has, list(...), "T")
    expect_error(mm(X = 1), "missing required argument")
    expect_error(mm(X = 1, y = 2, zzz = 3), "unknown argument")
    expect_error(mm(X = 1, y = 2, X = 3), "more than once")
    expect_error(mm(1, 2, 3, 4, 5), "too many arguments")
    # an ambiguous abbreviation is reported as ambiguous, not as a typo
    expect_error(mm(1, 2, pi = 3), "matches several arguments")
})

test_that("declared and Rcpp-reported spellings of a type compare equal", {
    n <- AI4BayesCode:::.ai4b_norm_type
    # the .cpp writes arma::vec; Rcpp reports the resolved arma::Col<double>
    expect_identical(n("const arma::vec&"), n("arma::Col<double>"))
    expect_identical(n("const arma::mat&"), n("arma::Mat<double>"))
    expect_false(identical(n("int"), n("const arma::vec&")))
    expect_false(identical(n("bool"), n("double")))
})

test_that("a class with several constructors resolves by the arguments given", {
    sgs <- sig_of("FiniteGaussianMixture.cpp", "FiniteGaussianMixture")
    expect_gte(length(sgs), 2L)             # simple + advanced explicit-hyper
    ty <- function(s, n) unname(s$types[seq_len(n)])
    bound <- list(list(nargs = 4L, types = ty(sgs[[1]], 4L)),
                  list(nargs = length(sgs[[2]]$names),
                       types = ty(sgs[[2]], length(sgs[[2]]$names))))
    res <- function(...) AI4BayesCode:::.ai4b_resolve_ctor_call(
        sgs, bound, list(...), "FiniteGaussianMixture")

    y <- matrix(0, 4, 2)
    # naming only simple-constructor arguments lands on the simple one
    expect_length(res(y = y, rng_seed = 7L), 4L)
    # naming an advanced hyperparameter lands on the advanced one
    adv <- res(y = y, K = 3L, mu_0 = c(0, 0), kappa_0 = 0.1,
               a_lambda_0 = 2, b_lambda_0 = 1, alpha_dir = 1, rng_seed = 7L)
    expect_length(adv, length(sgs[[2]]$names))
    # a name belonging to NO constructor is still an error
    expect_error(res(y = y, nonesuch = 1), "unknown argument")
})

test_that("every bundled example defaults everything except its data", {
    # A default declared only in the pybind11 binding would make the argument
    # optional in Python and mandatory in R; the C++ signature is the single
    # source both frontends read, so the R-side parse must already see it.
    dir <- ai4bayescode_examples_path()
    skip_if(!nzchar(dir) || !dir.exists(dir), "bundled examples not found")
    files <- list.files(dir, pattern = "\\.cpp$", full.names = TRUE)
    skip_if(!length(files), "bundled examples not found")
    offenders <- character(0)
    for (f in files) {
        src <- paste(readLines(f, warn = FALSE), collapse = "\n")
        for (cn in AI4BayesCode:::.ai4b_exposed_class_names(src)) {
            sgs <- AI4BayesCode:::.ai4b_ctor_signatures(f, cn)
            if (!length(sgs)) next
            s <- sgs[[1]]
            # rng_seed and keep_history are never data: they must be optional
            for (k in c("rng_seed", "keep_history")) {
                i <- which(s$names == k)
                if (length(i) == 1L && !s$has_default[i])
                    offenders <- c(offenders, sprintf("%s::%s", cn, k))
            }
        }
    }
    expect_identical(offenders, character(0))
})

test_that("no constructor parameter is swallowed by methods::new's own formal", {
    # new(Class, ...) matches its OWN `Class` formal before forwarding, so a
    # parameter whose name is a prefix of "Class" never reaches the object and
    # the call fails complaining about class definitions.
    dir <- ai4bayescode_examples_path()
    skip_if(!nzchar(dir) || !dir.exists(dir), "bundled examples not found")
    files <- list.files(dir, pattern = "\\.cpp$", full.names = TRUE)
    skip_if(!length(files), "bundled examples not found")
    offenders <- character(0)
    for (f in files) {
        src <- paste(readLines(f, warn = FALSE), collapse = "\n")
        for (cn in AI4BayesCode:::.ai4b_exposed_class_names(src))
            for (s in AI4BayesCode:::.ai4b_ctor_signatures(f, cn)) {
                bad <- s$names[startsWith("Class", s$names)]
                if (length(bad))
                    offenders <- c(offenders, sprintf("%s::%s", cn, paste(bad, collapse = ",")))
            }
    }
    expect_identical(offenders, character(0))
})

test_that("every example that walks history takes the last_draw_only switch", {
    # predict_at(new_data, last_draw_only = TRUE) is what lets a caller who
    # kept the history get ONE prediction out of it. It only works if the
    # wrapper branches on use_history; one that walks its history
    # unconditionally would return the whole history instead.
    dir <- ai4bayescode_examples_path()
    skip_if(!nzchar(dir) || !dir.exists(dir), "bundled examples not found")
    files <- list.files(dir, pattern = "\\.cpp$", full.names = TRUE)
    skip_if(!length(files), "bundled examples not found")

    # The body of predict_at, by brace matching -- a fixed-width slice runs
    # past the end of the function and picks up the constructor's own
    # keep_history_.
    predict_at_body <- function(src) {
        m <- regexpr("history_map[ \t\n]+predict_at[ \t\n]*\\(", src, perl = TRUE)
        if (m < 0) return("")
        ch <- strsplit(substring(src, m), "")[[1]]
        open_at <- which(ch == "{")[1L]
        if (is.na(open_at)) return("")
        depth <- 0L
        for (k in seq(open_at, length(ch))) {
            if (ch[k] == "{") depth <- depth + 1L
            else if (ch[k] == "}") {
                depth <- depth - 1L
                if (depth == 0L) return(paste(ch[open_at:k], collapse = ""))
            }
        }
        ""
    }

    offenders <- character(0)
    for (f in files) {
        src <- paste(readLines(f, warn = FALSE), collapse = "\n")
        if (!grepl("kernel_control_mixin", src, fixed = TRUE)) next   # ARDLasso
        body <- predict_at_body(src)
        if (!nzchar(body)) next
        if (!grepl("keep_history_", body, fixed = TRUE)) next         # no history branch
        if (!grepl("use_history", body, fixed = TRUE) ||
            !grepl("last_draw_only", body, fixed = TRUE))
            offenders <- c(offenders, basename(f))
    }
    expect_identical(offenders, character(0))
})

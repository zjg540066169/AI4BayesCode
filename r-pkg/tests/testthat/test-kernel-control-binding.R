# AI4BAYESCODE_BIND_KERNEL_CONTROL, compiled with ai4bayescode_source().
# clear_history is bound through kcm_detail::clear_history_binding: a wrapper
# whose own clear_history takes another parameter list, or is private, gets
# the mixin's (the stored draws are emptied and its own function is not
# called); a wrapper whose own clear_history takes no argument gets that one;
# a class derived from a wrapper that declares clear_history() gets that one.
# The other entries use static_cast, so a wrapper that redeclares a mixin
# forwarder with another signature does not compile. ARDLasso (no mixin)
# takes last_draw_only and the no-argument predict_at().
# These compile C++, so they are skipped on CRAN.

.kc_prelude <- r"---(
// [[Rcpp::depends(RcppArmadillo)]]
#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif
#include <RcppArmadillo.h>
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/gamma_gibbs_block.hpp"
#include "AI4BayesCode/rcpp_wrap.hpp"
#include "AI4BayesCode/kernel_control_mixin.hpp"
#include <memory>
#include <random>
#include <string>
#include <vector>

// Two gamma_gibbs blocks "s" and "t", keep_history on; own_calls counts the
// calls of a wrapper's own clear_history.
#define KC_TOY_BODY(CLS)                                                        \
public:                                                                         \
    CLS() : rng_(7), impl_(std::make_unique<AI4BayesCode::composite_block>(#CLS)) { \
        for (const char* nm : {"s", "t"}) {                                     \
            AI4BayesCode::gamma_gibbs_block_config c;                           \
            c.name = nm;                                                        \
            c.initial_value = 1.0;                                              \
            c.params_fn = [](const AI4BayesCode::block_context&) {             \
                return AI4BayesCode::gamma_params{3.0, 2.0};                    \
            };                                                                  \
            impl_->add_child(                                                   \
                std::make_unique<AI4BayesCode::gamma_gibbs_block>(std::move(c))); \
        }                                                                       \
        impl_->set_keep_history(true);                                          \
    }                                                                           \
    void step(int n) { for (int i = 0; i < n; ++i) impl_->step(rng_); }       \
    int n_rows() const { return static_cast<int>(impl_->get_history().at("s").n_rows); } \
    int calls() const { return own_calls; }                                     \
    AI4BayesCode::history_map predict_at(const AI4BayesCode::state_map&) const { \
        return {};                                                              \
    }                                                                           \
    int own_calls = 0;                                                          \
    std::vector<double> readapt_args;                                           \
    void readapt_NUTS(int n, bool reset, int depth, double target) {            \
        readapt_args = {double(n), double(reset), double(depth), target};       \
    }                                                                           \
    void readapt_NUTS(int n, bool reset = false, int depth = -1) {             \
        readapt_NUTS(n, reset, depth, -2.0);                                    \
    }                                                                           \
    std::vector<double> last_readapt() const { return readapt_args; }          \
    std::mt19937_64 rng_;                                                       \
    std::unique_ptr<AI4BayesCode::composite_block> impl_;
)---"

.kc_toys <- paste0(.kc_prelude, r"---(
class KcPlain : public AI4BayesCode::kernel_control_mixin<KcPlain> {
    KC_TOY_BODY(KcPlain)
};
class KcBoolClear : public AI4BayesCode::kernel_control_mixin<KcBoolClear> {
    KC_TOY_BODY(KcBoolClear)
    void clear_history(bool) { ++own_calls; }
};
class KcStringClear : public AI4BayesCode::kernel_control_mixin<KcStringClear> {
    KC_TOY_BODY(KcStringClear)
    std::string clear_history() { ++own_calls; return "own"; }
};
class KcPrivateClear : public AI4BayesCode::kernel_control_mixin<KcPrivateClear> {
    KC_TOY_BODY(KcPrivateClear)
private:
    void clear_history() { ++own_calls; }
};
class KcSub : public KcPlain {
public:
    std::vector<double> own_buf;
    void step(int n) {
        KcPlain::step(n);
        for (int i = 0; i < n; ++i) own_buf.push_back(1.0);
    }
    int own_rows() const { return static_cast<int>(own_buf.size()); }
    void clear_history() { KcPlain::clear_history(); own_buf.clear(); ++own_calls; }
};


// readapt_NUTS of other shapes, bound by AI4BAYESCODE_BIND_READAPT_NUTS /
// AI4BAYESCODE_PYBIND_READAPT_NUTS. last() is (n, reset, depth, target) of
// the last call; target -9 marks a readapt_NUTS without target_accept and -2
// the 3-argument overload of a class with both.
#define RX_BODY                                                                 \
public:                                                                         \
    mutable std::vector<double> args;                                           \
    std::vector<double> last() const { return args; }
class Rx4d {     // 1.1.2 compiled this one; 1.1.3 with static_cast did not
    RX_BODY
    void readapt_NUTS(int n, bool r, int d, double t = -1.0) { args = {double(n), double(r), double(d), t}; }
};
class Rx4 {
    RX_BODY
    void readapt_NUTS(int n, bool r, int d, double t) { args = {double(n), double(r), double(d), t}; }
};
class Rx3 {
    RX_BODY
    void readapt_NUTS(int n, bool r = false, int d = -1) { args = {double(n), double(r), double(d), -9.0}; }
};
class RxBothDef {  // the 3-argument call would be ambiguous
    RX_BODY
    void readapt_NUTS(int n, bool r, int d, double t = -1.0) { args = {double(n), double(r), double(d), t}; }
    void readapt_NUTS(int n, bool r = false, int d = -1) { args = {double(n), double(r), double(d), -2.0}; }
};
class RxConst {
    RX_BODY
    void readapt_NUTS(int n, bool r, int d, double t) const { args = {double(n), double(r), double(d), t}; }
};
class RxSizeT {
    RX_BODY
    void readapt_NUTS(std::size_t n, bool r, int d, double t) { args = {double(n), double(r), double(d), t}; }
};
class RxNone {   // takes neither argument list: compiles, every call raises
    RX_BODY
    void readapt_NUTS(int n) { args = {double(n)}; }
};

#define KC_BIND(CLS)                                                            \
    Rcpp::class_<CLS>(#CLS)                                                     \
        .constructor()                                                          \
        .method("step", static_cast<void (CLS::*)(int)>(&CLS::step))            \
        .method("n_rows", static_cast<int (CLS::*)() const>(&CLS::n_rows))      \
        .method("calls", static_cast<int (CLS::*)() const>(&CLS::calls))        \
        AI4BAYESCODE_BIND_KERNEL_CONTROL(CLS)

RCPP_MODULE(KcToysR_module) {
    KC_BIND(KcPlain)
        .method("last_readapt", &KcPlain::last_readapt)
        AI4BAYESCODE_BIND_READAPT_NUTS(KcPlain);
    KC_BIND(KcBoolClear);
    KC_BIND(KcStringClear);
    KC_BIND(KcPrivateClear);
    KC_BIND(KcSub).method("own_rows", &KcSub::own_rows);
#define RX_BIND(CLS)                                                            \
    Rcpp::class_<CLS>(#CLS)                                                     \
        .constructor()                                                          \
        .method("last", static_cast<std::vector<double> (CLS::*)() const>(&CLS::last)) \
        AI4BAYESCODE_BIND_READAPT_NUTS(CLS)
    RX_BIND(Rx4d); RX_BIND(Rx4); RX_BIND(Rx3); RX_BIND(RxBothDef);
    RX_BIND(RxConst); RX_BIND(RxSizeT); RX_BIND(RxNone);
}
)---")

# freeze_names redeclared with another parameter type: the static_cast of the
# freeze entry stops the build (a C-style cast compiled it and called the
# function with an argument of another type).
.kc_bad <- paste0(.kc_prelude, r"---(
class KcBadFreeze : public AI4BayesCode::kernel_control_mixin<KcBadFreeze> {
    KC_TOY_BODY(KcBadFreeze)
    void freeze_names(const std::vector<std::string>&) {}
};
RCPP_MODULE(KcBadR_module) {
    Rcpp::class_<KcBadFreeze>("KcBadFreeze")
        .constructor()
        AI4BAYESCODE_BIND_KERNEL_CONTROL(KcBadFreeze);
}
)---")

test_that("clear_history binding calls the wrapper's own no-argument clear_history, else the mixin's", {
    skip_on_cran()
    skip_if_not_installed("Rcpp")
    skip_if_not_installed("RcppArmadillo")
    e <- new.env()
    suppressMessages(ai4bayescode_source(.kc_toys, env = e, quiet = TRUE))
    cases <- list(KcPlain = c(0, 3), KcBoolClear = c(0, 3), KcPrivateClear = c(0, 3),
                  KcStringClear = c(1, 8), KcSub = c(1, 3))
    for (cls in names(cases)) {
        m <- methods::new(e[[cls]])
        m$step(5L)
        expect_identical(m$n_rows(), 5L)
        m$clear_history()
        m$step(3L)
        expect_identical(c(m$calls(), m$n_rows()), as.integer(cases[[cls]]), info = cls)
    }
    sub <- methods::new(e$KcSub)
    sub$step(4L)
    sub$clear_history()
    sub$step(2L)
    expect_identical(c(sub$n_rows(), sub$own_rows()), c(2L, 2L))
})

test_that("readapt_NUTS binding selects the 3- and 4-argument overloads", {
    skip_on_cran()
    skip_if_not_installed("Rcpp")
    skip_if_not_installed("RcppArmadillo")
    e <- new.env()
    suppressMessages(ai4bayescode_source(.kc_toys, env = e, quiet = TRUE))
    m <- methods::new(e$KcPlain)
    m$readapt_NUTS(5L, TRUE, 3L)
    expect_identical(m$last_readapt(), c(5, 1, 3, -2))
    m$readapt_NUTS(6L, FALSE, -1L, 0.8)
    expect_identical(m$last_readapt(), c(6, 0, -1, 0.8))
})

test_that("readapt_NUTS binding: a class with one of the two forms calls it", {
    skip_on_cran()
    skip_if_not_installed("Rcpp")
    skip_if_not_installed("RcppArmadillo")
    e <- new.env()
    suppressMessages(ai4bayescode_source(.kc_toys, env = e, quiet = TRUE))
    call3 <- function(cls) { m <- methods::new(e[[cls]]); m$readapt_NUTS(5L, TRUE, 3L); m$last() }
    call4 <- function(cls, t) { m <- methods::new(e[[cls]]); m$readapt_NUTS(6L, FALSE, -1L, t); m$last() }
    # only the 4-argument form: the 3-argument call passes target_accept = -1
    for (cls in c("Rx4d", "Rx4", "RxConst", "RxSizeT")) {
        expect_identical(call3(cls), c(5, 1, 3, -1), info = cls)
        expect_identical(call4(cls, 0.7), c(6, 0, -1, 0.7), info = cls)
    }
    # both forms, the 3-argument call ambiguous: each overload as before
    expect_identical(call3("RxBothDef"), c(5, 1, 3, -2))
    expect_identical(call4("RxBothDef", 0.7), c(6, 0, -1, 0.7))
    # only the 3-argument form: the 4-argument call with a target_accept that
    # keeps the block's target calls it; one in (0, 1] raises an error
    expect_identical(call3("Rx3"), c(5, 1, 3, -9))
    expect_identical(call4("Rx3", -1), c(6, 0, -1, -9))
    expect_error(call4("Rx3", 0.7), "takes no target_accept")
    # neither form
    expect_error(call3("RxNone"), "takes neither")
    expect_error(call4("RxNone", -1), "takes neither")
})

test_that("a redeclared mixin forwarder with another signature does not compile", {
    skip_on_cran()
    skip_if_not_installed("Rcpp")
    skip_if_not_installed("RcppArmadillo")
    e <- new.env()
    expect_error(suppressMessages(suppressWarnings(
        utils::capture.output(ai4bayescode_source(.kc_bad, env = e, quiet = TRUE),
                              type = "output"))))
    expect_false(exists("KcBadFreeze", envir = e, inherits = FALSE))
})

test_that("ARDLasso: predict_at(new_data, last_draw_only), predict_at(), matrix X", {
    skip_on_cran()
    skip_if_not_installed("Rcpp")
    skip_if_not_installed("RcppArmadillo")
    e <- new.env()
    suppressMessages(ai4bayescode_example("ARDLasso", env = e, quiet = TRUE))
    set.seed(5)
    X <- matrix(stats::rnorm(200), 50, 4)
    Y <- as.numeric(1 + X %*% c(2, 0, -1, 0) + 0.5 * stats::rnorm(50))
    Xn <- matrix(stats::rnorm(24), 6, 4)
    m <- methods::new(e$ARDLasso, X, Y, 9L, TRUE)
    m$step(12L)
    h <- m$get_history()
    full <- m$predict_at(list(X = Xn))
    expect_identical(dim(full$y_hat), c(12L, 6L))
    expect_equal(full$y_hat, h$beta %*% t(Xn) + as.vector(h$alpha), tolerance = 1e-12)
    last <- m$predict_at(list(X = Xn), TRUE)
    expect_identical(dim(last$y_hat), c(1L, 6L))
    expect_equal(last$y_hat, full$y_hat[12L, , drop = FALSE], tolerance = 1e-12)
    expect_identical(dim(m$predict_at(list(X = as.vector(Xn)), FALSE)$y_hat), c(12L, 6L))
    expect_identical(dim(m$predict_at()$y_rep), c(12L, 50L))
    expect_identical(names(m$predict_at()), names(m$predict_at(list())))
    expect_error(m$predict_at(list(X = Xn[, 1:3])), "has 3 columns but the model was fitted with p = 4")
})

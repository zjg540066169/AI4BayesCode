// Copyright (C) 2026 AI4BayesCode contributors. GPL-3.0-or-later. See LICENSE at the repository root.
// backend_neutral.hpp
//
// Backend-neutral helpers so a sampler's *shared* code (the parts compiled
// under BOTH the Rcpp (R) and pybind11 (Python) backends) does not have to
// name `Rcpp::` or `R::` symbols directly.
//
//   - ai4b::stop(...)     : raise an error. Under the Rcpp backend it forwards
//                           to Rcpp::stop (identical message/formatting and R
//                           error semantics); under the pybind/standalone
//                           backend it throws std::runtime_error, which
//                           pybind11 translates to a Python exception.
//   - ai4b::lgammafn(x)   : std::lgamma(x)  (replacement for R::lgammafn).
//   - ai4b::digamma(x)    : digamma / psi for x > 0 (replacement for
//                           R::digamma / Rf_digamma), validated against R.
//
// Include this from the shared region of a sampler .cpp. It pulls in nothing
// backend-specific on its own.

#ifndef AI4BAYESCODE_BACKEND_NEUTRAL_HPP
#define AI4BAYESCODE_BACKEND_NEUTRAL_HPP

#include <string>
#include <stdexcept>
#include <cmath>
#include <cstdio>
#include <utility>

namespace ai4b {

// ----------------------------------------------------------------------------
// stop(): backend-neutral error.
// ----------------------------------------------------------------------------
#if defined(AI4BAYESCODE_RCPP_MODULE) || defined(Rcpp_hpp) || defined(RcppArmadillo__RcppArmadillo__h)

// R backend: delegate to Rcpp::stop so the message formatting (tinyformat,
// "%d"/"%s"/...) and the resulting R condition are exactly as before.
[[noreturn]] inline void stop(const std::string& msg) { Rcpp::stop(msg); }

template <class... Args>
[[noreturn]] inline void stop(const char* fmt, Args&&... args) {
    Rcpp::stop(fmt, std::forward<Args>(args)...);
}

#else

// pybind / standalone backend: throw a std C++ exception (pybind11 maps
// std::runtime_error -> RuntimeError, std::invalid_argument -> ValueError).
[[noreturn]] inline void stop(const std::string& msg) {
    throw std::runtime_error(msg);
}

// printf-style overload for the few format-string call sites. NOTE: scalar
// arguments only (ints, doubles, const char*); under this backend it is used
// for plain shared-code validation messages. std::string arguments to "%s"
// are not supported here (pass `.c_str()` if ever needed).
template <class... Args>
[[noreturn]] inline void stop(const char* fmt, Args... args) {
    const int n = std::snprintf(nullptr, 0, fmt, args...);
    std::string s;
    if (n > 0) {
        s.resize(static_cast<std::size_t>(n));
        std::snprintf(&s[0], static_cast<std::size_t>(n) + 1, fmt, args...);
    } else {
        s = fmt;
    }
    throw std::runtime_error(s);
}

#endif

// ----------------------------------------------------------------------------
// warning(): backend-neutral warning (non-fatal). Mirrors stop() but does NOT
// terminate. Under the Rcpp backend it forwards to Rcpp::warning (same R
// warning condition / formatting); under pybind/standalone it prints to
// stderr.
// ----------------------------------------------------------------------------
#if defined(AI4BAYESCODE_RCPP_MODULE) || defined(Rcpp_hpp) || defined(RcppArmadillo__RcppArmadillo__h)

// R backend: delegate to Rcpp::warning so the message formatting and the
// resulting R warning condition are exactly as before.
inline void warning(const std::string& msg) { Rcpp::warning(msg); }

template <class... Args>
inline void warning(const char* fmt, Args&&... args) {
    Rcpp::warning(fmt, std::forward<Args>(args)...);
}

#else

// pybind / standalone backend: print to stderr (non-fatal).
inline void warning(const std::string& msg) {
    std::fprintf(stderr, "warning: %s\n", msg.c_str());
}

// printf-style overload for the few format-string call sites. NOTE: scalar
// arguments only (ints, doubles, const char*), same convention as stop()'s
// else-branch. std::string arguments to "%s" are not supported here (pass
// `.c_str()` if ever needed).
template <class... Args>
inline void warning(const char* fmt, Args... args) {
    const int n = std::snprintf(nullptr, 0, fmt, args...);
    std::string s;
    if (n > 0) {
        s.resize(static_cast<std::size_t>(n));
        std::snprintf(&s[0], static_cast<std::size_t>(n) + 1, fmt, args...);
    } else {
        s = fmt;
    }
    std::fprintf(stderr, "warning: %s\n", s.c_str());
}

#endif

// ----------------------------------------------------------------------------
// lgammafn(): log-gamma. R::lgammafn(x) == std::lgamma(x) for x > 0.
// ----------------------------------------------------------------------------
inline double lgammafn(double x) { return std::lgamma(x); }

// ----------------------------------------------------------------------------
// digamma(): psi(x) for x > 0. Recurrence up to x >= 6 then the standard
// asymptotic (Bernoulli) expansion; matches R's digamma() to ~1e-12.
// ----------------------------------------------------------------------------
inline double digamma(double x) {
    double result = 0.0;
    while (x < 6.0) {
        result -= 1.0 / x;
        x += 1.0;
    }
    const double inv  = 1.0 / x;
    const double inv2 = inv * inv;
    result += std::log(x) - 0.5 * inv
            - inv2 * (1.0 / 12.0
                      - inv2 * (1.0 / 120.0
                                - inv2 * (1.0 / 252.0)));
    return result;
}

}  // namespace ai4b

#endif  // AI4BAYESCODE_BACKEND_NEUTRAL_HPP

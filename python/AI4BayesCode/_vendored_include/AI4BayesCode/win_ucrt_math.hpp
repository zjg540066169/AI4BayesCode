// Copyright (C) 2026 AI4BayesCode contributors. GPL-3.0-or-later. See LICENSE at the repository root.
// win_ucrt_math.hpp
//
//  Windows: exp, log and pow from the Universal C Runtime.
//
//  Some MinGW-w64 toolchains (Rtools) link their static libmingwex versions
//  of these three, which run several times slower than the runtime's own; the
//  runtime's they reach only through its import table. The R and Python
//  packages compile every sampler on Windows with
//      -DAI4B_WIN_UCRT_MATH -include AI4BayesCode/win_ucrt_math.hpp
//  and link it with
//      -Wl,--wrap=exp,--wrap=log,--wrap=pow
//  which sends every call to the three (also those from static libraries
//  such as OpenBLAS) to the functions below. Each calls the runtime's version
//  through its import-table entry. They are inline and marked used, so every
//  C++ translation unit emits them and the linker keeps one copy. A C
//  translation unit gets nothing from this file.
// ============================================================================
#ifndef AI4BAYESCODE_WIN_UCRT_MATH_HPP
#define AI4BAYESCODE_WIN_UCRT_MATH_HPP

#if defined(_WIN32) && defined(AI4B_WIN_UCRT_MATH) && defined(__cplusplus)
extern "C" {
extern double (*__imp_exp)(double);
extern double (*__imp_log)(double);
extern double (*__imp_pow)(double, double);
inline __attribute__((used)) double __wrap_exp(double x) { return __imp_exp(x); }
inline __attribute__((used)) double __wrap_log(double x) { return __imp_log(x); }
inline __attribute__((used)) double __wrap_pow(double x, double y) { return __imp_pow(x, y); }
}
#endif

#endif  // AI4BAYESCODE_WIN_UCRT_MATH_HPP

/*
 *  bart_staging/src/bart_kernel_unity.h
 *
 *  Unity / amalgamation header for the unified BART tree kernel,
 *  ported from the legacy bart/bart_kernel_unity.h. Used by
 *  AI4BayesCode/bart_block.hpp so that any downstream sourceCpp of
 *  BartNoise.cpp (or similar BART-using wrapper outside bart_staging/)
 *  pulls in the BART kernel .cpp files into the same translation unit.
 *
 *  When source-cpp'ing build_bart.cpp directly from bart_staging/,
 *  Rcpp's auto-discovery of sibling .cpp files handles linking and
 *  this header is unnecessary. It is required for the wrapper-from-
 *  examples-dir use case where Rcpp does NOT walk into src/BART/.
 *
 *  Rcpp package builds (as opposed to sourceCpp) should NOT include
 *  this header; instead, list the individual .cpp files in src/ so
 *  the R build system compiles them separately. A macro guard is
 *  provided so you can opt out by defining AI4BAYESCODE_BART_NO_UNITY
 *  before including bart_block.hpp.
 */

#ifndef AI4BAYESCODE_BART_KERNEL_UNITY_H_
#define AI4BAYESCODE_BART_KERNEL_UNITY_H_

#ifndef AI4BAYESCODE_BART_NO_UNITY

// Order: rtnorm and rand_draws need randomkit's PRNG primitives, but
// randomkit.cpp defines `#define N 624` and `#define M 397` at global
// scope. Include the macro-polluting file LAST and #undef immediately
// so user code can still use N / M as variable names.

#include "BART/treefuns.cpp"
#include "BART/bartfuns.cpp"
#include "BART/lambda.cpp"
#include "BART/rand_draws.cpp"
#include "BART/rtnorm.cpp"
#include "BART/randomkit.cpp"

// Scrub randomkit's global-scope macros.
#ifdef N
# undef N
#endif
#ifdef M
# undef M
#endif

#endif // AI4BAYESCODE_BART_NO_UNITY

#endif // AI4BAYESCODE_BART_KERNEL_UNITY_H_

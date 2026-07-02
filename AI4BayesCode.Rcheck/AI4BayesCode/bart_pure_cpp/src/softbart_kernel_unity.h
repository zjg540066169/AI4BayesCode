/*
 *  bart/src/softbart_kernel_unity.h
 *
 *  Unity / amalgamation header for the vendored SoftBart kernel
 *  (SOFTBART_VENDOR/). Mirrors bart_kernel_unity.h for the standard
 *  BART kernel.
 *
 *  Why this exists
 *  ---------------
 *  The vendored SoftBart sources split declarations (soft_bart.h)
 *  from definitions (soft_bart_impl.h). Including only soft_bart.h
 *  leaves linker symbols like `logpdf_beta` unresolved, which fails
 *  Rcpp::sourceCpp() of any wrapper that pulls in softbart_model.h.
 *
 *  Including this unity header in the translation unit drags in
 *  soft_bart_impl.h exactly once, so all SoftBart symbols are
 *  defined locally. softbart_block.hpp pulls this in automatically
 *  so wrapper authors don't have to think about it.
 *
 *  When NOT to include
 *  -------------------
 *  R-package builds (NOT sourceCpp) compile each .cpp separately
 *  and link them together. In that case, soft_bart_impl.h should be
 *  included in exactly ONE .cpp file (typically a top-level
 *  build_softbart.cpp), and this header should not be pulled in
 *  from softbart_block.hpp. Opt out by defining
 *  AI4BAYESCODE_SOFTBART_NO_UNITY before #including softbart_block.hpp.
 */

#ifndef AI4BAYESCODE_SOFTBART_KERNEL_UNITY_H_
#define AI4BAYESCODE_SOFTBART_KERNEL_UNITY_H_

#ifndef AI4BAYESCODE_SOFTBART_NO_UNITY

#include "SOFTBART_VENDOR/soft_bart_impl.h"

#endif // AI4BAYESCODE_SOFTBART_NO_UNITY

#endif // AI4BAYESCODE_SOFTBART_KERNEL_UNITY_H_

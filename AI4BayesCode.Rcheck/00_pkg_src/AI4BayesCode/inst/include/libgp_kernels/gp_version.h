// libgp kernel subsystem — vendored into AI4BayesCode.
// Upstream: https://github.com/mblum/libgp (BSD-3)
// This file replaces the upstream gp_version.h which was auto-generated
// via CMake. We don't need dynamic-library visibility macros since we
// compile statically into the Rcpp unity translation unit.

#ifndef LIBGP_VERSION_H
#define LIBGP_VERSION_H

#define LIBGP_VERSION_MAJOR 0
#define LIBGP_VERSION_MINOR 1
#define LIBGP_VERSION_PATCH 0
#define LIBGP_VERSION "0.1.0-AI4BayesCode-vendored"

// No dynamic-library export/import — we're compiled statically.
#define LIBGP_EXPORT

#endif // LIBGP_VERSION_H

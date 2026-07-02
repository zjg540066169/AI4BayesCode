// libgp kernel subsystem — unity amalgamation header for AI4BayesCode.
//
// This header #includes all vendored libgp kernel .cc files so the
// user's single-file Rcpp::sourceCpp translation unit compiles them
// in-place. This matches the pattern used for bart/.
//
// Upstream: https://github.com/mblum/libgp (BSD-3 Clause)
// See COPYING in this directory for the full license.
//
// We vendor only libgp's KERNEL SUBSYSTEM (CovarianceFunction + concrete
// kernel classes + factory + sum/product composition). We intentionally
// skip libgp's GaussianProcess class, SampleSet, CG/RProp optimizers,
// and Python bindings because AI4BayesCode manages GP state itself via the
// shared_data + refresher pattern — see
// AI4BayesCode/include/AI4BayesCode/elliptical_slice_sampling_block.hpp and
// AI4BayesCode/examples/GPRegression.cpp for the integration pattern.

#ifndef AI4BAYESCODE_LIBGP_KERNELS_UNITY_H
#define AI4BAYESCODE_LIBGP_KERNELS_UNITY_H

// Headers first (declare the types we need publicly).
#include "cov.h"
#include "cov_factory.h"
#include "cov_noise.h"
#include "cov_se_iso.h"
#include "cov_se_ard.h"
#include "cov_matern3_iso.h"
#include "cov_matern5_iso.h"
#include "cov_linear_one.h"
#include "cov_linear_ard.h"
#include "cov_periodic.h"
#include "cov_periodic_matern3_iso.h"
#include "cov_rq_iso.h"
#include "cov_sum.h"
#include "cov_prod.h"
#include "input_dim_filter.h"
#include "gp_utils.h"

// Source .cc files — included directly so a single sourceCpp call
// picks up all symbols. Guarded to prevent double-inclusion if some
// user already includes a unity header upstream.

#ifndef AI4BAYESCODE_LIBGP_KERNELS_UNITY_IMPL
#define AI4BAYESCODE_LIBGP_KERNELS_UNITY_IMPL
#include "src/cov.cc"
#include "src/cov_factory.cc"
#include "src/cov_noise.cc"
#include "src/cov_se_iso.cc"
#include "src/cov_se_ard.cc"
#include "src/cov_matern3_iso.cc"
#include "src/cov_matern5_iso.cc"
#include "src/cov_linear_one.cc"
#include "src/cov_linear_ard.cc"
#include "src/cov_periodic.cc"
#include "src/cov_periodic_matern3_iso.cc"
#include "src/cov_rq_iso.cc"
#include "src/cov_sum.cc"
#include "src/cov_prod.cc"
#include "src/input_dim_filter.cc"
#include "src/gp_utils.cc"
#endif

#endif // AI4BAYESCODE_LIBGP_KERNELS_UNITY_H

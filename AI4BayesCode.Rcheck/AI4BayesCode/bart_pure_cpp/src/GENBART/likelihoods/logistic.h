/*
 *  Copyright (C) 2026 AI4BayesCode
 *  GPL v2 or later.
 */

/*
 * likelihoods/logistic.h -- Y_i ~ Bernoulli(sigma(lambda_i)), sigma=logistic
 *
 * This is the classification path; compares against BART pkg's Holmes-Held
 * augmentation (paper §4.1).  No nuisance parameters.
 *
 * Numerical notes:
 *   - log_f uses log1p(exp(-|lam|)) form to avoid overflow at |lam| > 36
 *   - Fisher info = sigma * (1 - sigma).  When |lam| is huge this goes to
 *     zero and the Laplace proposal variance explodes; the v_precision_lb
 *     in laplace_opts bounds it.
 *   - For extreme |lam| > 36 (typical after a single tree adds mass), we
 *     saturate sigma to avoid underflow in (y - sigma) * lam terms.
 */

#ifndef GENBART_LIK_LOGISTIC_H_
#define GENBART_LIK_LOGISTIC_H_

#include <cmath>
#include "../likelihood_interface.h"

namespace genbart { namespace lik {

// Numerically-safe sigmoid.  Returns value in (eps, 1 - eps) for |lam| < 37,
// and in [eps, 1 - eps] past that (clamped).
inline double safe_sigmoid(double lam) {
  // Clamp lam to prevent exp overflow/underflow in downstream use.
  const double L = 36.0;
  if (lam >  L) return 1.0 - 1e-16;
  if (lam < -L) return 1e-16;
  if (lam >= 0.0) {
    const double e = std::exp(-lam);
    return 1.0 / (1.0 + e);
  } else {
    const double e = std::exp(lam);
    return e / (1.0 + e);
  }
}

// log(1 + exp(lam)) without overflow.
inline double log1p_exp(double lam) {
  if (lam >  36.0) return lam;            // exp(lam) >> 1
  if (lam < -36.0) return std::exp(lam);  // exp(lam) ~ 0
  return std::log1p(std::exp(lam));
}

class logistic_lik : public likelihood {
public:
  // Y must be 0 or 1; no scaling.
  logistic_lik() = default;

  double log_f(double y, double lam, std::size_t /*obs_i*/) const override {
    return y * lam - log1p_exp(lam);
  }

  double score(double y, double lam, std::size_t /*obs_i*/) const override {
    return y - safe_sigmoid(lam);
  }

  double fisher_info(double lam) const override {
    const double s = safe_sigmoid(lam);
    return s * (1.0 - s);
  }

  double obs_info(double /*y*/, double lam, std::size_t /*obs_i*/) const override {
    const double s = safe_sigmoid(lam);
    return s * (1.0 - s);
  }

  const char* name() const override { return "logistic"; }
  std::size_t num_nuisance() const override { return 0; }
};

}}  // namespace genbart::lik

#endif  // GENBART_LIK_LOGISTIC_H_

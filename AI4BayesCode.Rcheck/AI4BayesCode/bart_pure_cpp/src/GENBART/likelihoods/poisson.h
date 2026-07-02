/*
 *  Copyright (C) 2026 AI4BayesCode
 *  GPL v2 or later.
 */

/*
 * likelihoods/poisson.h -- Y_i ~ Poisson(exp(lambda_i)), log-linear link.
 *
 * lambda_i = r(X_i) + offset_i, so lambda plays the role of log-rate.
 *
 * Numerical notes:
 *   - exp(lambda) overflows around lambda ~ 700 (double).  We clamp lambda
 *     to [-50, 50] inside exp() to prevent overflow while keeping relevant
 *     range (e^50 ~ 5e21, plenty for real applications).
 *   - For very negative lambda, exp(lambda) -> 0, which makes log_f -> y*lam
 *     (if y > 0 this is very negative, correctly indicating impossibility).
 */

#ifndef GENBART_LIK_POISSON_H_
#define GENBART_LIK_POISSON_H_

#include <cmath>
#include "../likelihood_interface.h"

namespace genbart { namespace lik {

// Clamped exp.
inline double safe_exp(double lam) {
  if (lam >  50.0) return std::exp(50.0);
  if (lam < -50.0) return 0.0;
  return std::exp(lam);
}

class poisson_lik : public likelihood {
public:
  poisson_lik() = default;

  // Kernel only: y*lam - exp(lam).  Omits -lgamma(y+1) which is a pure y-only
  // constant (cancels in MH ratios, shifted constant in Laplace log_post).
  // log_f_full() recovers the full density via log_y_const.
  double log_f(double y, double lam, std::size_t /*obs_i*/) const override {
    return y * lam - safe_exp(lam);
  }

  // Pure y-constant: -lgamma(y+1) = -log(y!).
  double log_y_const(double y, std::size_t obs_i) const override {
    if (!y_lgamma_.empty()) return -y_lgamma_.at(obs_i);  // use cache
    return -std::lgamma(y + 1.0);
  }

  // Cache lgamma(y+1) once when Y is set.
  void prepare(const std::vector<double>& y) override {
    y_lgamma_.resize(y.size());
    for (std::size_t i = 0; i < y.size(); ++i) {
      y_lgamma_[i] = std::lgamma(y[i] + 1.0);
    }
  }

  double score(double y, double lam, std::size_t /*obs_i*/) const override {
    return y - safe_exp(lam);
  }

  double fisher_info(double lam) const override {
    return safe_exp(lam);
  }

  double obs_info(double /*y*/, double lam, std::size_t /*obs_i*/) const override {
    return safe_exp(lam);
  }

  const char* name() const override { return "poisson"; }
  std::size_t num_nuisance() const override { return 0; }

private:
  std::vector<double> y_lgamma_;  // cache of lgamma(Y_i + 1)
};

}}  // namespace genbart::lik

#endif  // GENBART_LIK_POISSON_H_

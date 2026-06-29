/*
 *  Copyright (C) 2026 AI4BayesCode
 *  GPL v2 or later.
 */

/*
 * likelihoods/gamma_shape.h  --  Gamma shape regression (paper §4.4).
 *
 * Model:
 *     Y_i ~ Gamma(shape = alpha(X_i), rate = beta)
 *     alpha(X_i) = exp(r(X_i))
 *
 * This is the model where the SHAPE depends on X (and so can change the
 * shape of the hazard across observations), while `beta` is a global rate
 * nuisance parameter.
 *
 * Analytic forms from paper §4.4 (with a = exp(lambda)):
 *     log f(y | lam)  = a * log(beta) - lgamma(a) + (a - 1) log y - beta y
 *     U(y | lam)      = a * [log(beta) - digamma(a) + log(y)]
 *     I(lam)          = a^2 * trigamma(a)
 *
 * Fisher information is y-free (using E[log Y] = digamma(a) - log(beta)).
 *
 * Nuisance: beta.  Gibbs update with Gamma(a0, b0) prior:
 *     beta | rest ~ Gamma(a0 + sum_i alpha_i, b0 + sum_i Y_i).
 *
 * Numerical safety:
 *   - a = exp(lam) clamped at [1e-6, 1e6] to keep digamma/trigamma finite.
 *   - trigamma(a) for large a ~ 1/a, so I(lam) ~ a.  Always positive.
 */

#ifndef GENBART_LIK_GAMMA_SHAPE_H_
#define GENBART_LIK_GAMMA_SHAPE_H_

#include <cmath>
#include <vector>
#ifndef NoRcpp
#include <R.h>
#include <Rmath.h>
#endif
#include "../likelihood_interface.h"

namespace genbart { namespace lik {

class gamma_shape_lik : public likelihood {
public:
  explicit gamma_shape_lik(double beta_init = 1.0,
                           double a0 = 1.0,
                           double b0 = 1.0)
      : beta_(std::max(beta_init, 1e-6)), a0_(a0), b0_(b0) {}

  double beta() const { return beta_; }

  double log_f(double y, double lam, std::size_t /*obs_i*/) const override {
    if (y <= 0.0) return -std::numeric_limits<double>::infinity();
    const double a = safe_exp_shape(lam);
    // a*log(beta) - lgamma(a) + (a-1)*log(y) - beta*y
    return a * std::log(beta_) - std::lgamma(a)
         + (a - 1.0) * std::log(y) - beta_ * y;
  }

  double score(double y, double lam, std::size_t /*obs_i*/) const override {
    if (y <= 0.0) return 0.0;
    const double a = safe_exp_shape(lam);
    return a * (std::log(beta_) - R::digamma(a) + std::log(y));
  }

  double fisher_info(double lam) const override {
    const double a = safe_exp_shape(lam);
    return a * a * R::trigamma(a);
  }

  double obs_info(double /*y*/, double lam, std::size_t /*obs_i*/) const override {
    return fisher_info(lam);  // y-free for this parameterisation
  }

  // Gibbs update for beta | rest ~ Gamma(a0 + sum alpha_i, b0 + sum y_i).
  // Here we reconstruct alpha_i = exp(lam[i]).
  void update_nuisance(const std::vector<double>& y,
                       const std::vector<double>& lam,
                       rn& gen) override {
    const std::size_t N = y.size();
    double sum_alpha = 0.0, sum_y = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
      sum_alpha += safe_exp_shape(lam[i]);
      sum_y     += (y[i] > 0.0 ? y[i] : 0.0);
    }
    const double shape_post = a0_ + sum_alpha;
    const double rate_post  = b0_ + sum_y;
    if (rate_post <= 0.0 || !std::isfinite(shape_post)) return;
    const double b_new = gen.gamma(shape_post, rate_post);
    if (b_new > 0.0 && std::isfinite(b_new)) {
      beta_ = b_new;
      if (beta_ < 1e-6) beta_ = 1e-6;
    }
  }

  const char* name() const override { return "gamma_shape"; }
  std::size_t num_nuisance() const override { return 1; }
  std::vector<double> nuisance_snapshot() const override {
    return std::vector<double>{beta_};
  }

private:
  static double safe_exp_shape(double lam) {
    if (lam >  14.0) return std::exp(14.0);   // alpha ~ 1.2e6
    if (lam < -14.0) return std::exp(-14.0);  // alpha ~ 8e-7
    return std::exp(lam);
  }

  double beta_;
  double a0_, b0_;
};

}}  // namespace genbart::lik

#endif  // GENBART_LIK_GAMMA_SHAPE_H_

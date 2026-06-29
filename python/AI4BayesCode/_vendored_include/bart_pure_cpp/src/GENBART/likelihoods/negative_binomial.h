/*
 *  Copyright (C) 2026 AI4BayesCode
 *  GPL v2 or later.
 */

/*
 * likelihoods/negative_binomial.h -- Y_i ~ NegBin(mu_i, kappa)
 *
 *   mu_i = exp(r(X_i))       (log link)
 *   var(Y_i) = mu_i + mu_i^2 / kappa
 *
 * PMF (with m = exp(lambda)):
 *   log f(y | lambda, kappa)
 *     = lgamma(y + kappa) - lgamma(kappa) - lgamma(y + 1)
 *     + kappa * log(kappa) + y * lambda - (y + kappa) * log(m + kappa)
 *
 * Analytic derivatives:
 *   score      = y - (y + kappa) * m / (m + kappa)
 *   fisher_info(lambda) = m * kappa / (m + kappa)           (y-free, canonical)
 *   obs_info   = m * kappa * (y + kappa) / (m + kappa)^2    (y-dependent)
 *
 * For the Laplace proposal we use `fisher_info` (y-free, always positive).
 *
 * Nuisance: kappa (overdispersion).  RW-MH on log(kappa) with a Gamma(a0, b0)
 * prior on kappa.  Default (a0, b0) = (0.01, 0.01) -- very diffuse.
 *
 * Numerical safety:
 *   - m = safe_exp(lam) clamped to [0, exp(50)]
 *   - kappa lower-bounded at 1e-4
 *   - lgamma blows up for large y + kappa; clamp y or kappa as needed
 */

#ifndef GENBART_LIK_NEGATIVE_BINOMIAL_H_
#define GENBART_LIK_NEGATIVE_BINOMIAL_H_

#include <cmath>
#include "../likelihood_interface.h"

namespace genbart { namespace lik {

inline double safe_exp_nb(double lam) {
  if (lam >  50.0) return std::exp(50.0);
  if (lam < -50.0) return 0.0;
  return std::exp(lam);
}

class negative_binomial_lik : public likelihood {
public:
  explicit negative_binomial_lik(double kappa_init = 1.0,
                                 double prop_sd_log_kappa = 0.3,
                                 double prior_a_kappa = 0.01,
                                 double prior_b_kappa = 0.01)
      : kappa_(std::max(kappa_init, 1e-4)),
        prop_sd_(prop_sd_log_kappa),
        a0_(prior_a_kappa), b0_(prior_b_kappa) {}

  double kappa() const { return kappa_; }

  // Kernel: everything except -lgamma(y+1) (pure y-constant).
  double log_f(double y, double lam, std::size_t /*obs_i*/) const override {
    const double m = safe_exp_nb(lam);
    const double k = std::max(kappa_, 1e-4);
    return std::lgamma(y + k) - std::lgamma(k)
         + k * std::log(k) + y * lam - (y + k) * std::log(m + k);
  }

  double log_y_const(double y, std::size_t obs_i) const override {
    if (!y_lgamma_.empty()) return -y_lgamma_.at(obs_i);
    return -std::lgamma(y + 1.0);
  }

  void prepare(const std::vector<double>& y) override {
    y_lgamma_.resize(y.size());
    for (std::size_t i = 0; i < y.size(); ++i) y_lgamma_[i] = std::lgamma(y[i] + 1.0);
  }

  double score(double y, double lam, std::size_t /*obs_i*/) const override {
    const double m = safe_exp_nb(lam);
    const double k = std::max(kappa_, 1e-4);
    return y - (y + k) * m / (m + k);
  }

  double fisher_info(double lam) const override {
    const double m = safe_exp_nb(lam);
    const double k = std::max(kappa_, 1e-4);
    return m * k / (m + k);
  }

  double obs_info(double /*y*/, double lam, std::size_t /*obs_i*/) const override {
    // Use Fisher info for the Laplace proposal (y-free, canonical-link tradition).
    return fisher_info(lam);
  }

  // RW-MH on log(kappa) with Gamma(a0, b0) prior on kappa.
  void update_nuisance(const std::vector<double>& y,
                       const std::vector<double>& lam,
                       rn& gen) override {
    const double lk_cur = std::log(kappa_);
    const double lk_new = lk_cur + prop_sd_ * gen.normal();
    const double k_new  = std::exp(lk_new);
    if (k_new < 1e-4 || !std::isfinite(k_new)) return;

    double ll_cur = 0.0, ll_new = 0.0;
    const std::size_t N = y.size();
    for (std::size_t i = 0; i < N; ++i) {
      const double m = safe_exp_nb(lam[i]);
      ll_cur += std::lgamma(y[i] + kappa_) - std::lgamma(kappa_)
              + kappa_ * std::log(kappa_)
              - (y[i] + kappa_) * std::log(m + kappa_);
      ll_new += std::lgamma(y[i] + k_new)  - std::lgamma(k_new)
              + k_new  * std::log(k_new)
              - (y[i] + k_new)  * std::log(m + k_new);
    }
    // Gamma(a0, b0) prior: log p(kappa) = (a0-1) log kappa - b0 kappa (+ const)
    const double lp_cur = (a0_ - 1.0) * std::log(kappa_) - b0_ * kappa_;
    const double lp_new = (a0_ - 1.0) * std::log(k_new)  - b0_ * k_new;
    // Change of variables log(kappa): Jacobian = |d kappa / d log kappa| = kappa.
    // So log posterior on log-scale includes + log(kappa).
    const double log_r = (ll_new + lp_new + lk_new) - (ll_cur + lp_cur + lk_cur);
    if (std::isfinite(log_r) && std::log(gen.uniform()) < log_r) {
      kappa_ = k_new;
    }
  }

  const char* name() const override { return "negative_binomial"; }
  std::size_t num_nuisance() const override { return 1; }
  std::vector<double> nuisance_snapshot() const override {
    return std::vector<double>{kappa_};
  }

private:
  double kappa_;
  double prop_sd_;
  double a0_, b0_;
  std::vector<double> y_lgamma_;  // cache of lgamma(Y_i + 1)
};

}}  // namespace genbart::lik

#endif  // GENBART_LIK_NEGATIVE_BINOMIAL_H_

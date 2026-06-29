/*
 *  Copyright (C) 2026 AI4BayesCode
 *  GPL v2 or later.
 */

/*
 * likelihoods/heteroscedastic.h  --  Y_i ~ Normal(m_i, phi * V(m_i))
 *
 * with m_i = exp(lambda_i) (log link) and V(m) = m (Poisson-like variance
 * function).  Matches Linero (2022) §4.2 experiment: Y ~ Poisson(m)
 * modelled as Gaussian with mean-variance m = exp(r(X)).
 *
 * This is the marquee example where RJMCMC beats conjugate backfitting -- the
 * paper shows RMSE 3.25 vs rbart's 5.71 vs bartMachine's 6.67.
 *
 * Nuisance param: tau = 1/phi.  Conjugate Gamma update from
 *   tau | rest ~ Gamma(N/2, 0.5 sum_i (y_i - m_i)^2 / V(m_i)).
 *
 * Score and Fisher info derived in paper supplementary §S.3; for our
 * (g=exp, V=identity) choice they simplify to:
 *
 *   m = exp(lambda)
 *   U(y | lambda) = -1/2 + (y - m)^2 / (2 phi m) + (y - m) / phi
 *   I(lambda)     = 1/2 + m / phi     (expected information; y-free)
 *
 * Numerical notes:
 *   - safe_exp clamps lambda to [-50, 50] before exp.
 *   - I(lambda) -> 0 as m -> 0 at rate m/phi; Laplace precision clamp handles.
 *   - For phi -> 0 (over-confident), (y-m)^2/phi can blow up; we lower-bound
 *     phi at 1e-6 to avoid NaN in degenerate chains.
 */

#ifndef GENBART_LIK_HETEROSCEDASTIC_H_
#define GENBART_LIK_HETEROSCEDASTIC_H_

#include <cmath>
#include <algorithm>
#include "../likelihood_interface.h"

namespace genbart { namespace lik {

inline double safe_exp_het(double lam) {
  if (lam >  50.0) return std::exp(50.0);
  if (lam < -50.0) return 0.0;
  return std::exp(lam);
}

class heteroscedastic_lik : public likelihood {
public:
  // phi_init : initial dispersion; tau = 1/phi.
  // a0, b0   : Gamma prior on tau ~ Gamma(a0, b0) (so phi ~ Inv-Gamma(a0, b0))
  heteroscedastic_lik(double phi_init = 1.0, double a0 = 1.0, double b0 = 1.0)
      : phi_(std::max(phi_init, 1e-6)), a0_(a0), b0_(b0) {}

  // Clamp phi lower-bound on access to avoid div-by-zero in extreme chains.
  double phi() const { return std::max(phi_, 1e-6); }

  double log_f(double y, double lam, std::size_t /*obs_i*/) const override {
    const double m = safe_exp_het(lam);
    if (m <= 0.0) return -std::numeric_limits<double>::infinity();
    const double p = phi();
    const double V = m;                // V(m) = m
    const double r = y - m;
    return -0.5 * std::log(2.0 * M_PI * p * V) - 0.5 * r * r / (p * V);
  }

  double score(double y, double lam, std::size_t /*obs_i*/) const override {
    const double m = safe_exp_het(lam);
    if (m <= 0.0) return 0.0;
    const double p = phi();
    const double r = y - m;
    return -0.5 + r * r / (2.0 * p * m) + r / p;
  }

  double fisher_info(double lam) const override {
    const double m = safe_exp_het(lam);
    const double p = phi();
    return 0.5 + m / p;
  }

  double obs_info(double /*y*/, double lam, std::size_t /*obs_i*/) const override {
    // Use expected info (y-free, always positive).  Observed info is signed.
    return fisher_info(lam);
  }

  // Gibbs update for tau = 1/phi.
  // tau | rest ~ Gamma(a0 + N/2, b0 + 0.5 sum_i (y_i - m_i)^2 / V(m_i))
  void update_nuisance(const std::vector<double>& y,
                       const std::vector<double>& lam,
                       rn& gen) override {
    const std::size_t N = y.size();
    double rss_scaled = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
      const double m = safe_exp_het(lam[i]);
      if (m <= 0.0) continue;
      const double r = y[i] - m;
      rss_scaled += r * r / m;   // V(m) = m
    }
    const double shape = a0_ + 0.5 * (double)N;
    const double rate  = b0_ + 0.5 * rss_scaled;
    if (rate <= 0.0) return;  // numerical guard
    const double tau = gen.gamma(shape, rate);
    if (tau > 0.0 && std::isfinite(tau)) {
      phi_ = 1.0 / tau;
      if (phi_ < 1e-6) phi_ = 1e-6;
    }
  }

  const char* name() const override { return "heteroscedastic"; }
  std::size_t num_nuisance() const override { return 1; }
  std::vector<double> nuisance_snapshot() const override {
    return std::vector<double>{phi_};
  }

private:
  double phi_;
  double a0_, b0_;
};

}}  // namespace genbart::lik

#endif  // GENBART_LIK_HETEROSCEDASTIC_H_

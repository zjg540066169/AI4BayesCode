/*
 *  Copyright (C) 2026 AI4BayesCode
 *  GPL v2 or later.
 */

/*
 * likelihoods/normal.h -- Y_i ~ Normal(r(X_i), sigma^2)
 *
 * Nuisance parameter: sigma.  Conjugate prior: inv-chi-squared(nu, lambda)
 * (same convention as Chipman 2010 BART), so the full conditional for
 * sigma^2 is inverse-chi-squared with updated params.
 *
 * Analytic score and fisher_info are provided for speed; both are O(1) in
 * the observation, which matters because laplace_leaf() calls them n times.
 */

#ifndef GENBART_LIK_NORMAL_H_
#define GENBART_LIK_NORMAL_H_

#include <cmath>
#include "../likelihood_interface.h"

namespace genbart { namespace lik {

class normal_lik : public likelihood {
public:
  // Default prior on sigma^2: inverse-chi-squared with nu df and scale
  // such that P(sigma < sigma_hat) = quantile.  We let caller pass these
  // in directly; defaults mimic the BART package.
  normal_lik(double sigma_init = 1.0,
             double nu = 3.0,
             double lambda = 1.0)
      : sigma_(sigma_init), nu_(nu), lambda_(lambda) {}

  double log_f(double y, double lam, std::size_t /*obs_i*/) const override {
    const double z = (y - lam) / sigma_;
    return -0.5 * std::log(2.0 * M_PI) - std::log(sigma_) - 0.5 * z * z;
  }

  double score(double y, double lam, std::size_t /*obs_i*/) const override {
    return (y - lam) / (sigma_ * sigma_);
  }

  double fisher_info(double /*lam*/) const override {
    return 1.0 / (sigma_ * sigma_);
  }

  double obs_info(double /*y*/, double /*lam*/, std::size_t /*obs_i*/) const override {
    return 1.0 / (sigma_ * sigma_);
  }

  void leaf_sums(const double* y, const double* lam, const std::size_t* /*obs_idx*/,
                 std::size_t n,
                 double& sum_logf, double& sum_score, double& sum_info) const override {
    const double inv_s2 = 1.0 / (sigma_ * sigma_);
    const double c = -0.5 * std::log(2.0 * M_PI) - std::log(sigma_);
    sum_logf = sum_score = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      const double r = y[i] - lam[i];
      sum_logf  += c - 0.5 * r * r * inv_s2;
      sum_score += r * inv_s2;
    }
    sum_info = n * inv_s2;
  }

  // Gibbs step for sigma: inv-chi-squared(nu + N, (nu*lambda + RSS) / (nu + N)).
  void update_nuisance(const std::vector<double>& y,
                       const std::vector<double>& lam,
                       rn& gen) override {
    const std::size_t N = y.size();
    double rss = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
      const double r = y[i] - lam[i];
      rss += r * r;
    }
    // draw sigma^2 ~ inv-chi-square(nu + N, (nu*lambda + rss) / (nu + N))
    // <=> sigma = sqrt((nu*lambda + rss) / rchisq(nu + N))
    const double df = nu_ + (double)N;
    const double scale = nu_ * lambda_ + rss;
    const double chi2 = gen.chi_square(df);
    if (chi2 <= 0.0) return;   // numerical guard; keep old sigma_
    sigma_ = std::sqrt(scale / chi2);
  }

  const char* name() const override { return "normal"; }
  std::size_t num_nuisance() const override { return 1; }
  std::vector<double> nuisance_snapshot() const override {
    return std::vector<double>{sigma_};
  }

  // Inspection
  double sigma() const { return sigma_; }

private:
  double sigma_;
  double nu_;
  double lambda_;
};

}}  // namespace genbart::lik

#endif  // GENBART_LIK_NORMAL_H_

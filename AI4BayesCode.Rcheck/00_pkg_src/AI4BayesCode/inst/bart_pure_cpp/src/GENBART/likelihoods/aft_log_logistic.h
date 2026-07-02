/*
 *  Copyright (C) 2026 AI4BayesCode
 *  GPL v2 or later.
 */

/*
 * likelihoods/aft_log_logistic.h --  accelerated failure time model with
 *                                    logistic error (paper §4.3).
 *
 * Model:
 *     log T_i = r(X_i) + sigma * eps_i,     eps_i ~ Logistic(0, 1)
 *
 * Observed data:
 *     Y_i = min(T_i, C_i),  delta_i = I(T_i <= C_i)
 *
 * The y passed to log_f() is log(observed time Y_i).  The censoring
 * indicator delta is stored internally (set via set_delta()) and looked up
 * via obs_i.
 *
 * Likelihood per observation (with z = (log y - lambda)/sigma):
 *     log L_i = delta_i * [z - 2 log(1 + exp(z)) - log(sigma)]
 *             + (1 - delta_i) * [-log(1 + exp(z))]
 *             = delta_i * z - delta_i * log(sigma)
 *                            - (1 + delta_i) * log(1 + exp(z))
 *
 * Score (d log L / d lambda = -d log L / dz / sigma):
 *     U_i = (1 / sigma) * [(1 + delta_i) * s(z) - delta_i]
 *
 * Observed information (always non-negative):
 *     J_i = (1 / sigma^2) * (1 + delta_i) * s(z) * (1 - s(z))
 *
 * Nuisance:  sigma.  Prior: log(sigma) ~ Normal(0, prior_sd_log_sigma^2).
 * Updated via random-walk MH on log(sigma).
 *
 * Numerical safety:
 *   - z is clamped via safe log1p_exp (from logistic.h pattern) to avoid
 *     overflow when |z| is large.
 *   - s(z)(1 - s(z)) clamped at 1e-12 lower bound for Laplace stability.
 *   - sigma lower-bounded at 1e-4 to avoid division blow-up.
 */

#ifndef GENBART_LIK_AFT_LOG_LOGISTIC_H_
#define GENBART_LIK_AFT_LOG_LOGISTIC_H_

#include <cmath>
#include <stdexcept>
#include <vector>
#include "../likelihood_interface.h"
#include "logistic.h"   // for safe_sigmoid, log1p_exp

namespace genbart { namespace lik {

class aft_log_logistic_lik : public likelihood {
public:
  explicit aft_log_logistic_lik(double sigma_init = 1.0,
                                double prior_sd_log_sigma = 10.0,
                                double prop_sd_log_sigma  = 0.2)
      : sigma_(std::max(sigma_init, 1e-4)),
        prior_sd_(prior_sd_log_sigma),
        prop_sd_(prop_sd_log_sigma) {}

  // Set the delta vector (censoring indicators).  Must be called before the
  // model is used.  length matches N.
  void set_delta(const std::vector<double>& delta) {
    for (double d : delta) {
      if (d != 0.0 && d != 1.0)
        throw std::invalid_argument("aft_log_logistic: delta must be 0 or 1");
    }
    delta_ = delta;
  }

  double sigma() const { return sigma_; }

  // y = log(observed time).  lambda = r(X) + offset.  obs_i indexes into delta_.
  double log_f(double y, double lam, std::size_t obs_i) const override {
    const double s  = std::max(sigma_, 1e-4);
    const double z  = (y - lam) / s;
    const double d  = delta_.empty() ? 1.0 : delta_.at(obs_i);
    // delta * z - delta * log(sigma) - (1 + delta) * log(1 + exp(z))
    return d * z - d * std::log(s) - (1.0 + d) * log1p_exp(z);
  }

  double score(double y, double lam, std::size_t obs_i) const override {
    const double s  = std::max(sigma_, 1e-4);
    const double z  = (y - lam) / s;
    const double d  = delta_.empty() ? 1.0 : delta_.at(obs_i);
    const double sg = safe_sigmoid(z);
    return ((1.0 + d) * sg - d) / s;
  }

  double obs_info(double y, double lam, std::size_t obs_i) const override {
    const double s  = std::max(sigma_, 1e-4);
    const double z  = (y - lam) / s;
    const double d  = delta_.empty() ? 1.0 : delta_.at(obs_i);
    const double sg = safe_sigmoid(z);
    double w = sg * (1.0 - sg);
    if (w < 1e-12) w = 1e-12;
    return (1.0 + d) * w / (s * s);
  }

  // Random-walk MH on log(sigma).  Normal prior on log(sigma) with
  // scale `prior_sd_`.
  void update_nuisance(const std::vector<double>& y,
                       const std::vector<double>& lam,
                       rn& gen) override {
    if (delta_.empty()) return;  // not yet configured
    const double log_s_cur = std::log(sigma_);
    const double log_s_new = log_s_cur + prop_sd_ * gen.normal();
    const double s_new = std::exp(log_s_new);
    if (s_new < 1e-6 || !std::isfinite(s_new)) return;

    // Log-posterior delta: likelihood at sigma_new vs sigma_cur, plus prior.
    double lp_cur = 0.0, lp_new = 0.0;
    const double s_cur = sigma_;
    const std::size_t N = y.size();
    for (std::size_t i = 0; i < N; ++i) {
      const double d = delta_[i];
      const double zc = (y[i] - lam[i]) / s_cur;
      const double zn = (y[i] - lam[i]) / s_new;
      lp_cur += d * zc - d * std::log(s_cur) - (1.0 + d) * log1p_exp(zc);
      lp_new += d * zn - d * std::log(s_new) - (1.0 + d) * log1p_exp(zn);
    }
    // Prior: log_sigma ~ Normal(0, prior_sd_^2).
    lp_cur += -0.5 * log_s_cur * log_s_cur / (prior_sd_ * prior_sd_);
    lp_new += -0.5 * log_s_new * log_s_new / (prior_sd_ * prior_sd_);

    if (!std::isfinite(lp_new - lp_cur)) return;
    if (std::log(gen.uniform()) < lp_new - lp_cur) {
      sigma_ = s_new;
    }
  }

  const char* name() const override { return "aft_log_logistic"; }
  std::size_t num_nuisance() const override { return 1; }
  std::vector<double> nuisance_snapshot() const override {
    return std::vector<double>{sigma_};
  }

private:
  double              sigma_;
  double              prior_sd_;   // prior sd on log(sigma)
  double              prop_sd_;    // MH proposal sd on log(sigma)
  std::vector<double> delta_;
};

}}  // namespace genbart::lik

#endif  // GENBART_LIK_AFT_LOG_LOGISTIC_H_

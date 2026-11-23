/*
 *  Copyright (C) 2026 AI4BayesCode
 *  GPL v2 or later.
 */

/*
 * likelihoods/beta.h  --  Beta regression (proportion / rate outcomes).
 *
 * Model:
 *     Y_i ~ Beta(a_i, b_i),  a_i = mu_i * phi,  b_i = (1 - mu_i) * phi
 *     mu_i = sigma(r(X_i))      (logit link),   Y in (0, 1)
 *     phi  = precision > 0       (nuisance)
 *
 * Analytic derivatives (let s = sigma(lambda), mu = s, a = mu*phi, b = (1-mu)*phi):
 *     log f    = lgamma(phi) - lgamma(a) - lgamma(b)
 *              + (a - 1)*log(y) + (b - 1)*log(1 - y)
 *     dL/dmu   = phi * [digamma(b) - digamma(a) + log(y/(1-y))]
 *     U        = s*(1-s) * phi * [digamma(b) - digamma(a) + log(y/(1-y))]
 *     I(lam)   = (s*(1-s) * phi)^2 * (trigamma(a) + trigamma(b))
 *
 * Fisher info is y-free and always positive (trigamma > 0 for positive arg).
 *
 * Nuisance phi: random-walk MH on log(phi) with Normal prior on log(phi).
 *
 * Numerical safety:
 *   - Y clamped to [1e-10, 1 - 1e-10] for log(y), log(1-y).
 *   - mu = s(lam) clamped via safe_sigmoid from logistic.h.
 *   - phi lower-bounded at 1e-4.
 */

#ifndef GENBART_LIK_BETA_H_
#define GENBART_LIK_BETA_H_

#include <cmath>
#include <vector>
#ifndef NoRcpp
#include <R.h>
#include <Rmath.h>
#endif
#include "../likelihood_interface.h"
#include "logistic.h"  // safe_sigmoid

namespace genbart { namespace lik {

class beta_lik : public likelihood {
public:
  explicit beta_lik(double phi_init = 10.0,
                    double prior_sd_log_phi = 10.0,
                    double prop_sd_log_phi  = 0.2)
      : phi_(std::max(phi_init, 1e-4)),
        prior_sd_log_phi_(prior_sd_log_phi),
        prop_sd_log_phi_(prop_sd_log_phi) {}

  double phi() const { return phi_; }

  double log_f(double y, double lam, std::size_t obs_i) const override {
    // Clip y to (eps, 1-eps) to avoid -inf in log(y), log(1-y).
    double log_yc, log1m_yc;
    log_clipped_y(y, obs_i, log_yc, log1m_yc);
    const double s  = safe_sigmoid(lam);
    const double p  = phi_;
    const double a  = s * p;
    const double b  = (1.0 - s) * p;
    return std::lgamma(p) - std::lgamma(a) - std::lgamma(b)
         + (a - 1.0) * log_yc + (b - 1.0) * log1m_yc;
  }

  double score(double y, double lam, std::size_t obs_i) const override {
    double log_yc, log1m_yc;
    log_clipped_y(y, obs_i, log_yc, log1m_yc);
    const double s  = safe_sigmoid(lam);
    const double p  = phi_;
    const double a  = s * p;
    const double b  = (1.0 - s) * p;
    const double logit_y = log_yc - log1m_yc;
    const double dL_dmu = p * (R::digamma(b) - R::digamma(a) + logit_y);
    const double ds_dlam = s * (1.0 - s);
    return ds_dlam * dL_dmu;
  }

  double fisher_info(double lam) const override {
    const double s = safe_sigmoid(lam);
    const double p = phi_;
    const double a = s * p;
    const double b = (1.0 - s) * p;
    const double ds_dlam = s * (1.0 - s);
    return ds_dlam * ds_dlam * p * p * (R::trigamma(a) + R::trigamma(b));
  }

  double obs_info(double /*y*/, double lam, std::size_t /*obs_i*/) const override {
    return fisher_info(lam);   // y-free, always positive
  }

  // Nuisance phi: RW-MH on log(phi), Normal(0, prior_sd^2) prior on log(phi).
  void update_nuisance(const std::vector<double>& y,
                       const std::vector<double>& lam,
                       rn& gen) override {
    const double lp_cur = std::log(phi_);
    const double lp_new = lp_cur + prop_sd_log_phi_ * gen.normal();
    const double phi_new = std::exp(lp_new);
    if (phi_new < 1e-6 || !std::isfinite(phi_new)) return;

    // Compute full log-lik at both phi.
    double ll_cur = 0.0, ll_new = 0.0;
    const std::size_t N = y.size();
    for (std::size_t i = 0; i < N; ++i) {
      const double s_i = safe_sigmoid(lam[i]);
      double logy, log1my;
      log_clipped_y(y[i], i, logy, log1my);
      {
        const double a = s_i * phi_;
        const double b = (1.0 - s_i) * phi_;
        ll_cur += std::lgamma(phi_) - std::lgamma(a) - std::lgamma(b)
                + (a - 1.0) * logy + (b - 1.0) * log1my;
      }
      {
        const double a = s_i * phi_new;
        const double b = (1.0 - s_i) * phi_new;
        ll_new += std::lgamma(phi_new) - std::lgamma(a) - std::lgamma(b)
                + (a - 1.0) * logy + (b - 1.0) * log1my;
      }
    }
    // Prior.
    const double sd2 = prior_sd_log_phi_ * prior_sd_log_phi_;
    ll_cur += -0.5 * lp_cur * lp_cur / sd2;
    ll_new += -0.5 * lp_new * lp_new / sd2;

    if (!std::isfinite(ll_new - ll_cur)) return;
    if (std::log(gen.uniform()) < ll_new - ll_cur) phi_ = phi_new;
  }

  // Stores y_i, log(yc_i) and log(1 - yc_i) for log_f, score and
  // update_nuisance.
  void prepare(const std::vector<double>& y) override {
    y_ = y;
    log_yc_.resize(y.size());
    log1m_yc_.resize(y.size());
    for (std::size_t i = 0; i < y.size(); ++i) {
      const double yc = std::min(std::max(y[i], 1e-10), 1.0 - 1e-10);
      log_yc_[i] = std::log(yc);
      log1m_yc_[i] = std::log(1.0 - yc);
    }
  }

  const char* name() const override { return "beta"; }
  std::size_t num_nuisance() const override { return 1; }
  std::vector<double> nuisance_snapshot() const override {
    return std::vector<double>{phi_};
  }

private:
  // log(yc) and log(1 - yc), yc = y clipped to [1e-10, 1 - 1e-10]: the stored
  // values when y is the prepared y of obs_i, else computed.
  void log_clipped_y(double y, std::size_t obs_i,
                     double& log_yc, double& log1m_yc) const {
    if (obs_i < y_.size() && y_[obs_i] == y) {
      log_yc = log_yc_[obs_i];
      log1m_yc = log1m_yc_[obs_i];
      return;
    }
    const double yc = std::min(std::max(y, 1e-10), 1.0 - 1e-10);
    log_yc = std::log(yc);
    log1m_yc = std::log(1.0 - yc);
  }

  double phi_;
  double prior_sd_log_phi_;
  double prop_sd_log_phi_;
  std::vector<double> y_, log_yc_, log1m_yc_;  // prepared y_i and its logs
};

}}  // namespace genbart::lik

#endif  // GENBART_LIK_BETA_H_

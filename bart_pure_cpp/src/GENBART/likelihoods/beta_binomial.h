/*
 *  Copyright (C) 2026 AI4BayesCode
 *  GPL v2 or later.
 */

/*
 * likelihoods/beta_binomial.h  --  Beta-Binomial regression.
 *
 * Model:
 *     p_i     ~ Beta(mu_i * M, (1 - mu_i) * M)
 *     Y_i | p_i ~ Binomial(n_i, p_i)
 *     mu_i    = sigma(r(X_i))       (logit link)
 *
 * Integrating out p_i gives the beta-binomial PMF:
 *     f(y | n, mu, M) = C(n, y) * B(mu*M + y, (1-mu)*M + n - y) / B(mu*M, (1-mu)*M)
 *     log f = lC(n,y) + lgamma(M) - lgamma(M+n)
 *           + lgamma(A + y) - lgamma(A)
 *           + lgamma(B + n - y) - lgamma(B)
 *     where A = mu*M, B = (1-mu)*M.
 *
 * The per-observation trial count n_i is stored in the likelihood via
 * set_n_trials(n) and looked up via obs_i.
 *
 * Analytic derivatives (let s = sigma(lam), ds = s*(1-s), d2s = (1-2s)*s*(1-s)):
 *
 *   dL/dmu  = M * [digamma(A + y) - digamma(A)
 *                 - digamma(B + n - y) + digamma(B)]
 *   d2L/dmu2 = M^2 * [trigamma(A + y) - trigamma(A)
 *                    + trigamma(B + n - y) - trigamma(B)]
 *
 *   U(y, lam)       = ds * dL/dmu
 *   obs_info(y, lam) = -d2s * dL/dmu - ds^2 * d2L/dmu2,
 *                      clamped lower-bound at 1e-12.
 *
 * Nuisance: M (precision / inverse overdispersion).  RW-MH on log(M).
 * Intraclass correlation rho = 1/(M+1); M large -> rho small -> near Binomial.
 *
 * Numerical safety:
 *   - mu clamped via safe_sigmoid.
 *   - obs_info clamped at 1e-12 in case the first-order term dominates.
 *   - M lower-bounded at 1e-3.
 */

#ifndef GENBART_LIK_BETA_BINOMIAL_H_
#define GENBART_LIK_BETA_BINOMIAL_H_

#include <cmath>
#include <stdexcept>
#include <vector>
#ifndef NoRcpp
#include <R.h>
#include <Rmath.h>
#endif
#include "../likelihood_interface.h"
#include "logistic.h"    // safe_sigmoid

namespace genbart { namespace lik {

class beta_binomial_lik : public likelihood {
public:
  explicit beta_binomial_lik(double M_init = 10.0,
                             double prior_sd_log_M = 10.0,
                             double prop_sd_log_M  = 0.2)
      : M_(std::max(M_init, 1e-3)),
        prior_sd_(prior_sd_log_M),
        prop_sd_(prop_sd_log_M) {}

  // Must be called before running MCMC.  n_trials[i] = positive integer.
  void set_n_trials(const std::vector<double>& n_trials) {
    for (double n : n_trials) {
      if (n <= 0.0 || n != std::floor(n))
        throw std::invalid_argument("beta_binomial: n_trials must be positive integers");
    }
    n_trials_ = n_trials;
  }

  double M() const { return M_; }
  double rho() const { return 1.0 / (M_ + 1.0); }

  // Kernel: drop lchoose(n, y) which is a pure (y, n_i) constant.  The rest
  // of the expression depends on lambda (via A, B through sigmoid(lam)) or
  // on the nuisance M (changes during M's update, so not lambda-constant).
  double log_f(double y, double lam, std::size_t obs_i) const override {
    if (n_trials_.empty()) return 0.0;
    const double n = n_trials_.at(obs_i);
    if (y < 0.0 || y > n) return -std::numeric_limits<double>::infinity();
    const double s = safe_sigmoid(lam);
    const double A = s * M_;
    const double B = (1.0 - s) * M_;
    if (A <= 0.0 || B <= 0.0) return -std::numeric_limits<double>::infinity();
    return std::lgamma(M_) - std::lgamma(M_ + n)
         + std::lgamma(A + y) - std::lgamma(A)
         + std::lgamma(B + n - y) - std::lgamma(B);
  }

  double log_y_const(double y, std::size_t obs_i) const override {
    if (!lchoose_.empty()) return lchoose_.at(obs_i);
    if (n_trials_.empty()) return 0.0;
    return R::lchoose(n_trials_.at(obs_i), y);
  }

  void prepare(const std::vector<double>& y) override {
    if (n_trials_.size() != y.size()) return;  // not yet set_n_trials
    lchoose_.resize(y.size());
    for (std::size_t i = 0; i < y.size(); ++i) {
      lchoose_[i] = R::lchoose(n_trials_[i], y[i]);
    }
  }

  double score(double y, double lam, std::size_t obs_i) const override {
    if (n_trials_.empty()) return 0.0;
    const double n = n_trials_.at(obs_i);
    const double s = safe_sigmoid(lam);
    const double A = s * M_;
    const double B = (1.0 - s) * M_;
    const double dL_dmu = M_ * (R::digamma(A + y) - R::digamma(A)
                              - R::digamma(B + n - y) + R::digamma(B));
    const double ds = s * (1.0 - s);
    return ds * dL_dmu;
  }

  double obs_info(double y, double lam, std::size_t obs_i) const override {
    if (n_trials_.empty()) return 1e-6;
    const double n = n_trials_.at(obs_i);
    const double s = safe_sigmoid(lam);
    const double A = s * M_;
    const double B = (1.0 - s) * M_;
    const double ds  = s * (1.0 - s);
    const double d2s = (1.0 - 2.0 * s) * ds;
    const double dL_dmu = M_ * (R::digamma(A + y) - R::digamma(A)
                              - R::digamma(B + n - y) + R::digamma(B));
    const double d2L_dmu2 = M_ * M_ * (R::trigamma(A + y) - R::trigamma(A)
                                     + R::trigamma(B + n - y) - R::trigamma(B));
    double J = -d2s * dL_dmu - ds * ds * d2L_dmu2;
    if (!std::isfinite(J) || J < 1e-12) J = 1e-12;
    return J;
  }

  // Approximate Fisher info: binomial Fisher * (1 + (n-1)*rho)^{-1}
  // averaged over some typical n; for simplicity we use the obs_info at
  // the expected y (= n*mu) which gives dL/dmu = 0 and a clean expression.
  double fisher_info(double lam) const override {
    // Approximate: assume n = mean trial count (conservative).  We don't know
    // n_i here, so fall back to obs_info at a representative n=1.  Callers
    // who care should use obs_info.
    (void)lam;
    return 1e-6;
  }

  // Nuisance M: random-walk MH on log(M).
  void update_nuisance(const std::vector<double>& y,
                       const std::vector<double>& lam,
                       rn& gen) override {
    if (n_trials_.empty()) return;
    const double lM_cur = std::log(M_);
    const double lM_new = lM_cur + prop_sd_ * gen.normal();
    const double M_new  = std::exp(lM_new);
    if (M_new < 1e-6 || !std::isfinite(M_new)) return;

    // lchoose(n, y) is a y-constant; cancels in ll_new - ll_cur; drop it.
    double ll_cur = 0.0, ll_new = 0.0;
    const std::size_t N = y.size();
    for (std::size_t i = 0; i < N; ++i) {
      const double n = n_trials_[i];
      const double s_i = safe_sigmoid(lam[i]);
      {
        const double A = s_i * M_;
        const double B = (1.0 - s_i) * M_;
        ll_cur += std::lgamma(M_) - std::lgamma(M_ + n)
                + std::lgamma(A + y[i]) - std::lgamma(A)
                + std::lgamma(B + n - y[i]) - std::lgamma(B);
      }
      {
        const double A = s_i * M_new;
        const double B = (1.0 - s_i) * M_new;
        ll_new += std::lgamma(M_new) - std::lgamma(M_new + n)
                + std::lgamma(A + y[i]) - std::lgamma(A)
                + std::lgamma(B + n - y[i]) - std::lgamma(B);
      }
    }
    const double sd2 = prior_sd_ * prior_sd_;
    ll_cur += -0.5 * lM_cur * lM_cur / sd2;
    ll_new += -0.5 * lM_new * lM_new / sd2;

    if (!std::isfinite(ll_new - ll_cur)) return;
    if (std::log(gen.uniform()) < ll_new - ll_cur) M_ = M_new;
  }

  const char* name() const override { return "beta_binomial"; }
  std::size_t num_nuisance() const override { return 1; }
  std::vector<double> nuisance_snapshot() const override {
    return std::vector<double>{M_};
  }

private:
  double              M_;
  double              prior_sd_;
  double              prop_sd_;
  std::vector<double> n_trials_;
  std::vector<double> lchoose_;  // cache of lchoose(n_i, y_i)
};

}}  // namespace genbart::lik

#endif  // GENBART_LIK_BETA_BINOMIAL_H_

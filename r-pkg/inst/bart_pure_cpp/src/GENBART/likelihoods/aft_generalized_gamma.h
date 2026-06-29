/*
 *  Copyright (C) 2026 AI4BayesCode
 *  GPL v2 or later.
 */

/*
 * likelihoods/aft_generalized_gamma.h  --  AFT with generalized gamma error
 *                                          (paper §4.3).
 *
 * Model:
 *     log T_i = r(X_i) + sigma * eps_i,
 *     eps_i  ~ log Gam(alpha, alpha)
 *              i.e. eps = log X with X ~ Gamma(shape = alpha, rate = alpha)
 *
 * So f_eps(z) = (alpha^alpha / Gamma(alpha)) * exp(alpha*z - alpha*e^z).
 * The family interpolates between Weibull (alpha = 1) and log-Normal
 * (alpha -> infinity), making it useful for model discrimination.
 *
 * Observed data:
 *     y = log(Y),    delta = I(T <= C)
 *
 * Likelihood per observation with z = (y - lambda)/sigma:
 *     log L = delta * [log f_eps(z) - log sigma] + (1 - delta) * log S_eps(z)
 *
 *     log f_eps(z)  = alpha*log(alpha) - lgamma(alpha) + alpha*z - alpha*e^z
 *     log S_eps(z)  = log P(log X > z)  where X ~ Gam(alpha, rate=alpha)
 *                   = log(pgamma(alpha*e^z, alpha, alpha, lower = FALSE))
 *
 * Per paper §4.3, the generalized gamma model uses numerical derivatives;
 * we do the same here, using central finite differences from the base class
 * default for score and obs_info, driven by our analytic log_f.
 *
 * Nuisance:
 *   sigma  : RW-MH on log(sigma), Normal prior on log(sigma).
 *   alpha  : RW-MH on log(alpha), Uniform(0, alpha_max) prior.
 *            (paper uses Uniform(0, 40); we follow.)
 *
 * Numerical safety:
 *   - z clamped via log1p_exp-style tricks when computing exp(z).
 *   - pgamma evaluated on log scale via R::pgamma(..., log_p = TRUE).
 *   - sigma lower-bounded at 1e-4, alpha in [1e-2, alpha_max].
 */

#ifndef GENBART_LIK_AFT_GGAMMA_H_
#define GENBART_LIK_AFT_GGAMMA_H_

#include <cmath>
#include <stdexcept>
#include <vector>
#ifndef NoRcpp
#include <R.h>
#include <Rmath.h>
#endif
#include "../likelihood_interface.h"

namespace genbart { namespace lik {

class aft_generalized_gamma_lik : public likelihood {
public:
  explicit aft_generalized_gamma_lik(double sigma_init = 1.0,
                                     double alpha_init = 1.0,
                                     double alpha_max  = 40.0,
                                     double prior_sd_log_sigma = 10.0,
                                     double prop_sd_log_sigma  = 0.15,
                                     double prop_sd_log_alpha  = 0.2)
      : sigma_(std::max(sigma_init, 1e-4)),
        alpha_(std::max(std::min(alpha_init, alpha_max), 1e-2)),
        alpha_max_(alpha_max),
        prior_sd_log_sigma_(prior_sd_log_sigma),
        prop_sd_log_sigma_(prop_sd_log_sigma),
        prop_sd_log_alpha_(prop_sd_log_alpha) {}

  void set_delta(const std::vector<double>& delta) {
    for (double d : delta) {
      if (d != 0.0 && d != 1.0)
        throw std::invalid_argument("aft_generalized_gamma: delta must be 0 or 1");
    }
    delta_ = delta;
  }

  double sigma() const { return sigma_; }
  double alpha() const { return alpha_; }

  // Core log_f.  obs_i indexes delta_.
  double log_f(double y, double lam, std::size_t obs_i) const override {
    return log_f_at_params(y, lam, obs_i, sigma_, alpha_);
  }

  // Fallback defaults for score and obs_info (finite-difference from log_f).

  // Nuisance Gibbs: random-walk MH on log(sigma), log(alpha) sequentially.
  void update_nuisance(const std::vector<double>& y,
                       const std::vector<double>& lam,
                       rn& gen) override {
    if (delta_.empty()) return;

    // -- sigma step --
    {
      const double ls_cur = std::log(sigma_);
      const double ls_new = ls_cur + prop_sd_log_sigma_ * gen.normal();
      const double s_new  = std::exp(ls_new);
      if (std::isfinite(s_new) && s_new > 1e-6) {
        const double dll = full_loglik_delta(y, lam, s_new, alpha_)
                         - full_loglik_delta(y, lam, sigma_, alpha_);
        const double dprior =
            -0.5 * ls_new * ls_new / (prior_sd_log_sigma_ * prior_sd_log_sigma_)
          + 0.5 * ls_cur * ls_cur / (prior_sd_log_sigma_ * prior_sd_log_sigma_);
        if (std::isfinite(dll + dprior) &&
            std::log(gen.uniform()) < dll + dprior) {
          sigma_ = s_new;
        }
      }
    }

    // -- alpha step --
    {
      const double la_cur = std::log(alpha_);
      const double la_new = la_cur + prop_sd_log_alpha_ * gen.normal();
      const double a_new  = std::exp(la_new);
      if (std::isfinite(a_new) && a_new >= 1e-2 && a_new <= alpha_max_) {
        const double dll = full_loglik_delta(y, lam, sigma_, a_new)
                         - full_loglik_delta(y, lam, sigma_, alpha_);
        // Uniform(1e-2, alpha_max) prior in alpha space.  On log-alpha the
        // Jacobian factor log(alpha) cancels on log-uniform interval.
        // We add a log(a_new/alpha_cur) from the change-of-variables.
        const double dprior = la_new - la_cur;
        if (std::isfinite(dll + dprior) &&
            std::log(gen.uniform()) < dll + dprior) {
          alpha_ = a_new;
        }
      }
    }
  }

  const char* name() const override { return "aft_generalized_gamma"; }
  std::size_t num_nuisance() const override { return 2; }
  std::vector<double> nuisance_snapshot() const override {
    return std::vector<double>{sigma_, alpha_};
  }

private:
  // Parameterised log-likelihood evaluated at arbitrary sigma, alpha.
  // Used both by log_f (with current params) and update_nuisance.
  double log_f_at_params(double y, double lam, std::size_t obs_i,
                         double s, double a) const {
    const double z = (y - lam) / std::max(s, 1e-6);
    const double d = delta_.empty() ? 1.0 : delta_.at(obs_i);

    if (d > 0.5) {
      // Observed event: log f_eps(z) - log sigma.
      return a * std::log(a) - std::lgamma(a) + a * z - a * safe_exp_clip(z)
             - std::log(std::max(s, 1e-6));
    } else {
      // Censored: log S_eps(z) = log pgamma(a*exp(z), a, rate=a, lower=F, log=T).
      const double arg = a * safe_exp_clip(z);
      // R::pgamma(q, shape, scale, lower_tail, give_log).  We want rate=a
      // -> scale = 1/a.
      const double log_S = R::pgamma(arg, a, 1.0 / a, /*lower=*/0, /*log=*/1);
      if (!std::isfinite(log_S)) return -std::numeric_limits<double>::infinity();
      return log_S;
    }
  }

  double full_loglik_delta(const std::vector<double>& y,
                           const std::vector<double>& lam,
                           double s, double a) const {
    double total = 0.0;
    const std::size_t N = y.size();
    for (std::size_t i = 0; i < N; ++i) {
      const double lf = log_f_at_params(y[i], lam[i], i, s, a);
      if (!std::isfinite(lf)) return -std::numeric_limits<double>::infinity();
      total += lf;
    }
    return total;
  }

  static double safe_exp_clip(double z) {
    // exp(z) clamped to avoid overflow.  In AFT context z rarely exceeds ~30
    // before log_S saturates, so a hard cap at 50 is plenty.
    if (z >  50.0) return std::exp(50.0);
    if (z < -50.0) return 0.0;
    return std::exp(z);
  }

  double sigma_;
  double alpha_;
  double alpha_max_;
  double prior_sd_log_sigma_;
  double prop_sd_log_sigma_;
  double prop_sd_log_alpha_;
  std::vector<double> delta_;
};

}}  // namespace genbart::lik

#endif  // GENBART_LIK_AFT_GGAMMA_H_

/*
 *  Copyright (C) 2026 AI4BayesCode
 *  GPL v2 or later.  See LICENSE for full text.
 */

/*
 * likelihood_interface.h  --  abstract callback interface for generalized BART.
 *
 * Following Linero (2022), the user plugs in a likelihood family
 *
 *     Y_i ~ f_eta(Y_i | lambda_i)        where lambda_i = r(X_i) + offset_i
 *
 * by providing log f, its score U = d log f / d lambda, and an information
 * function (observed or Fisher) used to build the Laplace Gaussian proposal
 * for leaf parameters.
 *
 * Interface design notes:
 *
 *   - Core per-observation methods take (y, lambda, obs_i) where obs_i is
 *     the GLOBAL training observation index (0-based).  This lets AFT /
 *     BetaBin / etc. look up per-observation metadata (delta indicator,
 *     trial count) stored in subclass state.  Likelihoods that don't need
 *     per-obs metadata simply ignore obs_i (`(void)obs_i`).
 *
 *   - `obs_info(y, lambda, obs_i)` is the primary information source for
 *     the Laplace proposal because it can depend on the observed y (needed
 *     for AFT with censoring etc.).  Models with y-free observed information
 *     (Normal, Logistic, Poisson, Het) simply return a y-free constant.
 *
 *   - `fisher_info(lambda)` remains available for models that expose it;
 *     currently unused by the default Laplace path but kept for diagnostics
 *     and possible alternative samplers.
 *
 * Numerical safety:
 *
 *   - Use RELATIVE finite-difference step: h = max(1e-5 * |lambda|, 1e-6).
 *   - Fisher / observed information should be LOWER BOUNDED by the caller
 *     (via `laplace_opts::v_precision_lb`), not the likelihood, because the
 *     lower bound depends on the prior scale.
 *   - log_f MUST return -inf (not NaN) off-support.
 */

#ifndef GENBART_LIKELIHOOD_INTERFACE_H_
#define GENBART_LIKELIHOOD_INTERFACE_H_

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "BART/rn.h"

namespace genbart {

// ---------------------------------------------------------------------------
// Abstract base class.  Users subclass and override log_f() (required)
// and typically score() / obs_info() (strongly recommended for speed).
// All per-observation methods are const so they are thread-safe.
// Nuisance-parameter updates happen in update_nuisance(), which is NOT const.
// ---------------------------------------------------------------------------
class likelihood {
public:
  virtual ~likelihood() = default;

  // ===== Core: log-likelihood of observation (y, lambda, obs_i) =====
  // Must return a finite double on the support, or -infinity off-support.
  // Must NEVER return NaN.  obs_i is the global training-observation index.
  //
  // Implementers may return the FULL log-density or just the parameter-
  // dependent kernel -- both are valid.  If you implement only the kernel
  // here, override log_y_const() too if you want log_f_full() to be correct.
  virtual double log_f(double y, double lambda, std::size_t obs_i) const = 0;

  // ===== OPTIONAL: y-only / obs-only normalising constant =====
  // Default 0.  Override if your log_f OMITS a constant that depends only on
  // y / obs_i (e.g. -lgamma(y+1) in Poisson) and you want log_f_full() to be
  // correct for reporting purposes.
  virtual double log_y_const(double y, std::size_t obs_i) const {
    (void)y; (void)obs_i; return 0.0;
  }

  // ===== CONVENIENCE: log_f + log_y_const =====
  double log_f_full(double y, double lambda, std::size_t obs_i) const {
    return log_f(y, lambda, obs_i) + log_y_const(y, obs_i);
  }

  // ===== Score d log_f / d lambda =====
  // Default: central finite difference with relative step.
  virtual double score(double y, double lambda, std::size_t obs_i) const {
    const double h = fd_step(lambda);
    const double fp = log_f(y, lambda + h, obs_i);
    const double fm = log_f(y, lambda - h, obs_i);
    if (!std::isfinite(fp) || !std::isfinite(fm)) return 0.0;
    return (fp - fm) / (2.0 * h);
  }

  // ===== Observed information: -d^2 log_f / d lambda^2 at (y, lambda) =====
  // Default: central finite difference of log_f.
  // This is the PRIMARY information used by the Laplace proposal.
  virtual double obs_info(double y, double lambda, std::size_t obs_i) const {
    const double h = fd_step(lambda);
    const double fp = log_f(y, lambda + h, obs_i);
    const double f0 = log_f(y, lambda,      obs_i);
    const double fm = log_f(y, lambda - h, obs_i);
    if (!std::isfinite(fp) || !std::isfinite(f0) || !std::isfinite(fm)) return 0.0;
    return -(fp - 2.0 * f0 + fm) / (h * h);
  }

  // ===== Expected Fisher info I(lambda) (y-free) =====
  // Kept for diagnostics and for likelihoods that have a closed form.  NOT
  // used by the default Laplace path.  Default: throws (use obs_info).
  virtual double fisher_info(double lambda) const {
    (void)lambda;
    throw std::runtime_error("likelihood::fisher_info not implemented");
  }

  // ===== Batch versions (for speed) =====
  // Default: naive loop.  Override to vectorise.
  // NOTE: this is called only for diagnostic purposes; the Laplace leaf
  // routine uses per-observation calls (with obs_idx) for flexibility.
  virtual void leaf_sums(const double*       y,
                         const double*       lambda,
                         const std::size_t*  obs_idx,
                         std::size_t         n,
                         double&             sum_logf,
                         double&             sum_score,
                         double&             sum_info) const
  {
    sum_logf = sum_score = sum_info = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      const std::size_t idx = obs_idx ? obs_idx[i] : i;
      sum_logf  += log_f(y[i],   lambda[i], idx);
      sum_score += score(y[i],   lambda[i], idx);
      sum_info  += obs_info(y[i], lambda[i], idx);
    }
  }

  // ===== One-off setup when Y is set =====
  // Called by genbart_model on construction and on every set_Y.  Override to
  // precompute per-observation constants (e.g. -lgamma(y+1) in Poisson,
  // lchoose(n, y) in Beta-Binomial).  Default: no-op.
  virtual void prepare(const std::vector<double>& y) { (void)y; }

  // ===== Nuisance-parameter Gibbs/MH step =====
  // Called once per outer MCMC iteration, after all trees are updated.
  // `lambda[i]` holds CURRENT ensemble prediction r(X_i) + offset_i.
  virtual void update_nuisance(const std::vector<double>& y,
                                const std::vector<double>& lambda,
                                rn& gen) {
    (void)y; (void)lambda; (void)gen;
  }

  // ===== Inspection =====
  virtual const char* name() const = 0;
  virtual std::size_t num_nuisance() const { return 0; }
  virtual std::vector<double> nuisance_snapshot() const {
    return std::vector<double>();
  }

protected:
  // Relative-step finite-difference helper.  h = max(rel*|lambda|, abs).
  static double fd_step(double lambda) {
    constexpr double rel = 1e-5;
    constexpr double abs_min = 1e-6;
    const double h = rel * std::fabs(lambda);
    return h > abs_min ? h : abs_min;
  }
};

// ---------------------------------------------------------------------------
// Utility: clamp information to a safe lower bound to prevent runaway
// Laplace proposal variance.
// ---------------------------------------------------------------------------
inline double safe_info(double I, double eps_min) {
  if (!std::isfinite(I) || I < eps_min) return eps_min;
  return I;
}

}  // namespace genbart

#endif  // GENBART_LIKELIHOOD_INTERFACE_H_

/*
 *  Copyright (C) 2026 AI4BayesCode
 *  GPL v2 or later; see LICENSE.
 */

/*
 * laplace.h -- Laplace approximation of a leaf-parameter full-conditional,
 *              used to build the Gaussian proposal G_BIRTH / G_DEATH /
 *              G_CHANGE in Linero (2022) Algorithm 3.
 *
 * Given observations (y_i, lambda_i) assigned to a leaf, and a leaf prior
 *
 *     pi_mu(mu) = Normal(mu | 0, sigma_mu^2),
 *
 * we approximate the full conditional pi(mu | ...) ~ Normal(m, v^2) where
 *
 *     m      = argmax_{mu}  sum_i log f_eta(y_i | lambda_i + mu) + log pi_mu(mu)
 *     v^{-2} = sum_i obs_info_eta(y_i, lambda_i + m; obs_i) + sigma_mu^{-2}
 *
 * m is found by Fisher scoring (over the observed information for maximum
 * compatibility with non-canonical links -- for canonical / information-matrix-
 * equals-hessian cases, obs_info == Fisher info anyway).
 *
 * The routine takes an optional `obs_idx` array so likelihoods with per-
 * observation metadata (AFT delta, BetaBin n_i) can look up their state.
 * Pass nullptr if the likelihood doesn't need it.
 *
 * Numerical safeguards:
 *   (1) Step-halving on every Newton step that decreases the log-posterior.
 *   (2) Max-iteration cap + gradient-relative tolerance.
 *   (3) Lower bound on posterior precision to prevent runaway variance.
 *   (4) NaN / inf guards; failed Fisher scoring falls back to the prior.
 */

#ifndef GENBART_LAPLACE_H_
#define GENBART_LAPLACE_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

#include "likelihood_interface.h"

namespace genbart {

struct laplace_proposal {
  double m;         // posterior mode
  double v;         // posterior sd (sqrt of 1 / precision)
  double log_post;  // log p(mu=m | rest) up to constants (for diagnostics)
  bool   converged;
  bool   fallback;  // true = returned the prior as the proposal
  int    iters;
};

struct laplace_opts {
  // Paper Algorithm 3 + warm-start (parent mu / sibling-avg init) typically
  // converges in 1-3 iterations; max_iter=8 is a safe upper bound that saves
  // substantial time in the ~0.5% of cases where the old default 20 would
  // have iterated past convergence with no benefit.
  int    max_iter       = 8;
  // Paper Algorithm 3 uses |U| > sqrt(I) / 10 as the stopping condition,
  // equivalent to grad_tol = 0.1.  This is a LOOSE tolerance -- the paper
  // notes that "we do not need to compute (m_l, v_l) exactly, as we just
  // want reasonable Gaussian approximations to the full conditional; any
  // inaccuracies are naturally corrected for by their effect on the
  // Metropolis-Hastings acceptance probability."  Setting it tighter
  // costs a lot of Fisher-scoring iterations for negligible gain.
  double grad_tol       = 0.1;
  int    halving_max    = 10;
  double v_precision_lb = 1e-8;
  // Paper Algorithm 3:
  //   BIRTH moves: initialise at parent-node mu (the leaf being split)
  //   DEATH moves: initialise at (mu_lL + mu_lR) / 2
  // Callers pass the appropriate init_mu; default 0.0 for generic use.
  double init_mu        = 0.0;
  bool   verbose        = false;
};

// ---------------------------------------------------------------------------
// Core: Laplace proposal for a single leaf.
//
//   y_leaf     : y values in this leaf, length n
//   lam_leaf   : lambda_minus_t values, length n
//   obs_idx    : global observation indices, length n (OR nullptr if the
//                likelihood doesn't use per-obs metadata)
//   n          : leaf size
//   sigma_mu   : leaf prior sd
//   lik        : likelihood (const)
// ---------------------------------------------------------------------------
inline laplace_proposal laplace_leaf(
    const double*       y_leaf,
    const double*       lam_leaf,
    const std::size_t*  obs_idx,
    std::size_t         n,
    double              sigma_mu,
    const likelihood&   lik,
    laplace_opts        opts = laplace_opts())
{
  laplace_proposal out;
  out.converged = false;
  out.fallback  = false;
  out.iters     = 0;

  const double sigma_mu2 = sigma_mu * sigma_mu;
  const double inv_sigma_mu2 = 1.0 / sigma_mu2;

  if (n == 0) {
    out.m = 0.0;
    out.v = sigma_mu;
    out.log_post = 0.0;
    out.converged = true;
    return out;
  }

  auto idx_of = [&](std::size_t i) -> std::size_t {
    return obs_idx ? obs_idx[i] : i;
  };

  auto log_post_at = [&](double mu) {
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      const double lf = lik.log_f(y_leaf[i], lam_leaf[i] + mu, idx_of(i));
      if (!std::isfinite(lf)) return -std::numeric_limits<double>::infinity();
      s += lf;
    }
    s += -0.5 * mu * mu * inv_sigma_mu2;
    return s;
  };

  auto grad_info_at = [&](double mu, double& gr, double& info) {
    gr = 0.0;
    info = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      const std::size_t gi = idx_of(i);
      const double lam_i = lam_leaf[i] + mu;
      const double si = lik.score(y_leaf[i], lam_i, gi);
      const double Ii = lik.obs_info(y_leaf[i], lam_i, gi);
      if (!std::isfinite(si) || !std::isfinite(Ii)) { gr = info = NAN; return; }
      gr   += si;
      info += Ii;
    }
    gr   += -mu * inv_sigma_mu2;
    info += inv_sigma_mu2;
  };

  double mu = opts.init_mu;
  double lp_curr = log_post_at(mu);
  if (!std::isfinite(lp_curr)) {
    out.m = 0.0;
    out.v = sigma_mu;
    out.log_post = lp_curr;
    out.fallback = true;
    return out;
  }

  for (int iter = 0; iter < opts.max_iter; ++iter) {
    double gr, info;
    grad_info_at(mu, gr, info);
    if (!std::isfinite(gr) || !std::isfinite(info) || info <= 0.0) {
      out.m = 0.0;
      out.v = sigma_mu;
      out.log_post = lp_curr;
      out.fallback = true;
      out.iters = iter;
      return out;
    }

    if (std::fabs(gr) < opts.grad_tol * std::sqrt(info)) {
      out.converged = true;
      out.iters = iter;
      break;
    }

    double step = gr / info;
    double mu_new = mu + step;
    double lp_new = log_post_at(mu_new);
    int    halves = 0;
    while ((!std::isfinite(lp_new) || lp_new < lp_curr - 1e-12) &&
           halves < opts.halving_max)
    {
      step *= 0.5;
      mu_new = mu + step;
      lp_new = log_post_at(mu_new);
      ++halves;
    }

    if (halves >= opts.halving_max) {
      out.iters = iter + 1;
      break;
    }

    mu = mu_new;
    lp_curr = lp_new;
    out.iters = iter + 1;
  }

  double gr_final, info_final;
  grad_info_at(mu, gr_final, info_final);
  if (!std::isfinite(info_final) || info_final < opts.v_precision_lb) {
    info_final = opts.v_precision_lb;
  }

  out.m = mu;
  out.v = 1.0 / std::sqrt(info_final);
  out.log_post = lp_curr;
  return out;
}

// Log-density of Normal(m, v^2) at x.  Computed in log-space for safety.
inline double log_dnorm(double x, double m, double v) {
  constexpr double LOG_SQRT_2PI = 0.9189385332046727;
  const double z = (x - m) / v;
  return -LOG_SQRT_2PI - std::log(v) - 0.5 * z * z;
}

// Sample mu ~ Normal(m, v^2).
inline double rnorm_from_laplace(const laplace_proposal& lp, rn& gen) {
  return lp.m + lp.v * gen.normal();
}

}  // namespace genbart

#endif  // GENBART_LAPLACE_H_

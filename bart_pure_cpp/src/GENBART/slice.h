/*
 *  Copyright (C) 2026 AI4BayesCode
 *  GPL v2 or later; see LICENSE.
 */

/*
 * slice.h -- Neal (2003) univariate slice sampler for a leaf-parameter
 *            full conditional, implementing Linero (2022) Algorithm 2
 *            line 6: "Sample M_t targeting its full conditional using
 *            (say) slice sampling (Neal, 2003)."
 *
 * Given observations (y_i, lambda_i) assigned to a leaf, and the leaf prior
 *
 *     pi_mu(mu) = Normal(mu | 0, sigma_mu^2),
 *
 * one call performs ONE slice-sampling transition whose invariant
 * distribution is the EXACT full conditional
 *
 *     pi(mu | ...) propto pi_mu(mu) * prod_i f_eta(y_i | lambda_i + mu).
 *
 * This replaces the former uncorrected Laplace-approximate draw in the
 * M-step of rjmcmc_bd.h: a Gaussian draw from the Laplace approximation
 * without a Metropolis correction targets the conditional only
 * approximately, whereas the paper's exactness argument requires the
 * refresh to leave the full conditional invariant. A single slice
 * transition per leaf per sweep does exactly that (Alg 2 line 6 requires
 * invariance, not an independent draw -- slice sampling itself is a
 * Markov transition).
 *
 * Implementation choices (paper does not pin them down):
 *   - step width w = sigma_mu: state-independent (required for the
 *     stepping-out correctness proof) and an upper bound on the
 *     conditional scale for log-concave likelihoods, so stepping out is
 *     rarely needed and shrinkage costs O(log2(w / sd)) evaluations.
 *   - stepping-out cap m = 10 with Neal's J/K randomised split, which
 *     preserves detailed balance under a bounded expansion.
 *   - shrinkage loop capped at 100; on pathological non-termination the
 *     sampler holds the current value (staying put is always a valid
 *     transition and introduces no bias).
 *   - log_f returning -inf (off-support contract of likelihood_interface)
 *     simply lands the proposal outside the slice and shrinks.
 */

#ifndef GENBART_SLICE_H_
#define GENBART_SLICE_H_

#include <cmath>
#include <cstddef>
#include <limits>

#include "likelihood_interface.h"
#include "laplace.h"   // log_dnorm
#include "BART/rn.h"

namespace genbart {

/*
 * One Neal (2003, Fig. 3 + Fig. 5) slice-sampling transition for a single
 * leaf parameter.
 *
 *   ys, lams : the n observations (y_i, lambda_i) routed to this leaf
 *   idx      : per-observation indices forwarded to lik.log_f (may carry
 *              AFT delta / BetaBin n_i metadata); same convention as
 *              laplace_leaf.
 *   sigma_mu : sd of the Normal(0, sigma_mu^2) leaf prior.
 *   mu0      : current leaf value (the chain state; slice starts here).
 *
 * Returns the new leaf value. Never throws; on non-finite log density at
 * the current state (cannot happen for a valid chain state) or shrinkage
 * non-termination it returns mu0 unchanged.
 */
inline double slice_leaf(
    const double*       ys,
    const double*       lams,
    const std::size_t*  idx,
    std::size_t         n,
    double              sigma_mu,
    const likelihood&   lik,
    double              mu0,
    rn&                 gen,
    int                 max_stepout = 10,
    int                 max_shrink  = 100)
{
  const double neg_inf = -std::numeric_limits<double>::infinity();

  // log of the (unnormalised) full conditional.
  auto logg = [&](double mu) -> double {
    double lp = log_dnorm(mu, 0.0, sigma_mu);
    for (std::size_t j = 0; j < n; ++j) {
      lp += lik.log_f(ys[j], lams[j] + mu, idx[j]);
      if (!std::isfinite(lp)) return neg_inf;
    }
    return lp;
  };

  const double g0 = logg(mu0);
  if (!std::isfinite(g0)) return mu0;  // defensive: hold (valid transition)

  // 1. Slice level: log y = g(mu0) + log U.
  const double logy = g0 + std::log(gen.uniform());

  // 2. Stepping out (Neal Fig. 3), width w = sigma_mu, cap max_stepout
  //    with the randomised J/K budget split.
  const double w = sigma_mu;
  double L = mu0 - w * gen.uniform();
  double R = L + w;
  int J = static_cast<int>(std::floor(max_stepout * gen.uniform()));
  int K = (max_stepout - 1) - J;
  while (J > 0 && logg(L) > logy) { L -= w; --J; }
  while (K > 0 && logg(R) > logy) { R += w; --K; }

  // 3. Shrinkage (Neal Fig. 5): sample uniformly on [L, R], shrink toward
  //    mu0 on rejection. Terminates a.s.; capped defensively.
  for (int it = 0; it < max_shrink; ++it) {
    const double mu1 = L + gen.uniform() * (R - L);
    if (logg(mu1) > logy) return mu1;
    if (mu1 < mu0) L = mu1; else R = mu1;
  }
  return mu0;  // guard: hold current value (no bias)
}

}  // namespace genbart

#endif  // GENBART_SLICE_H_

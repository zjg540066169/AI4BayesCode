/*================================================================================
 *  AI4BayesCode: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v2.0 or later
 *  (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
 *================================================================================
 *
 *  bnp_utils.hpp -- header-only Bayesian-nonparametric (BNP) utility
 *                   functions used by stick_breaking_block, the truncated-
 *                   SBP example wrappers (DPGaussianMixture / PYGaussian-
 *                   Mixture / ...), and any user-written Neal-Algorithm-2/8
 *                   composition that needs CRP / PY weights or counts.
 *
 *  WHY HEADER-ONLY
 *  ===============
 *  These are short, pure functions used both inside library blocks and
 *  inside user-written log_probs_fn / refresher lambdas in `examples/*.cpp`.
 *  Header-only avoids needing a separate .cpp + linking step in
 *  Rcpp::sourceCpp builds.
 *
 *  CONTENTS
 *  --------
 *  - counts_from_z(z, K)            : histogram of cluster assignments
 *  - crp_log_prior(...)             : CRP allocation log-prior at a given k
 *  - py_log_prior(...)              : Pitman-Yor variant
 *  - crp_sample_new_assignment(...) : draw a fresh z_i from CRP weights
 *  - py_sample_new_assignment(...)  : Pitman-Yor variant
 *
 *  All functions live in namespace AI4BayesCode::bnp.
 *
 *  ALLOCATION CONVENTION
 *  ---------------------
 *  Cluster labels are 1-indexed in the public AI4BayesCode API (matches R /
 *  Stan / categorical_gibbs_block). Internal arithmetic in this file is
 *  0-indexed; the conversion happens at the boundaries.
 *
 *  All "counts" inputs are length K (the truncation level for SBP-based
 *  samplers, or the current number of populated clusters for CRP-marginal
 *  samplers).
 *================================================================================*/

#ifndef AI4BAYESCODE_BNP_UTILS_HPP
#define AI4BAYESCODE_BNP_UTILS_HPP

#include "block_sampler.hpp"  // brings in arma + namespace

#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

namespace AI4BayesCode {
namespace bnp {

// ----------------------------------------------------------------------------
// counts_from_z
// ----------------------------------------------------------------------------

/**
 * @brief Histogram of 1-indexed cluster assignments.
 *
 * Given a length-N vector z of cluster labels in {1, ..., K} (stored as
 * doubles for shared_data compatibility; non-integer values are rounded
 * to the nearest integer and clamped), returns an arma::vec of length K
 * whose k-th entry counts the number of i with z_i = k+1 (0-indexed
 * here; the k=0 entry is the count of label 1, etc.).
 *
 * Out-of-range labels (< 1 or > K after rounding) trigger an exception
 * — silently dropping them would give a wrong sufficient statistic.
 */
inline arma::vec counts_from_z(const arma::vec& z, std::size_t K) {
    if (K == 0) {
        throw std::invalid_argument("bnp::counts_from_z: K must be > 0");
    }
    arma::vec counts(K, arma::fill::zeros);
    for (std::size_t i = 0; i < z.n_elem; ++i) {
        const long lab = static_cast<long>(std::llround(z[i]));
        if (lab < 1 || static_cast<std::size_t>(lab) > K) {
            throw std::out_of_range(
                "bnp::counts_from_z: label out of range {1, ..., K}");
        }
        counts[static_cast<std::size_t>(lab) - 1] += 1.0;
    }
    return counts;
}

// ----------------------------------------------------------------------------
// Escobar-West (1995) DP-concentration update
// ----------------------------------------------------------------------------

/**
 * @brief Sample the DP concentration alpha by the Escobar & West (1995)
 *        auxiliary-variable Gibbs step, conditioning ONLY on the sufficient
 *        statistic (k, n).
 *
 * Escobar, M. D. & West, M. (1995), "Bayesian Density Estimation and
 * Inference Using Mixtures", JASA 90(430):577-588, Section 6 / equation
 * (13)-(14). With a Gamma(a, b) [shape-rate] prior on alpha, the full
 * conditional of alpha given (k, n) -- where k is the number of OCCUPIED
 * clusters and n is the number of observations -- depends on the rest of
 * the model ONLY through (k, n). It is sampled with a two-step auxiliary
 * scheme:
 *
 *     eta | alpha, n   ~  Beta(alpha + 1, n)
 *     alpha | eta, k   ~  w * Gamma(a + k,     b - log eta)
 *                       + (1 - w) * Gamma(a + k - 1, b - log eta)
 *
 * with the mixing weight defined by
 *
 *     w / (1 - w)  =  (a + k - 1) / ( n * (b - log eta) ).
 *
 * WHY THIS BEATS THE STICK-LEVEL UPDATE
 * -------------------------------------
 * In a TRUNCATED stick-breaking representation (Ishwaran & James 2001),
 * the "obvious" alpha conditional sums log(1 - V_k) over ALL K_trunc - 1
 * sticks, including the EMPTY-TAIL sticks V_k ~ Beta(1, alpha) that are
 * pure prior draws carrying no data information. Those tail draws differ
 * between chains and make alpha's conditional chain-specific, producing a
 * STRUCTURAL (not ESS-limited) non-mixing failure of rank-R-hat(alpha).
 *
 * Conditioning on (k, n) -- the genuinely data-identified sufficient
 * statistic for alpha under the DP -- severs that empty-stick coupling.
 * k and n are shared across chains once the allocation z mixes, so alpha
 * mixes too. This is the standard Antoniak (1974) / Escobar-West (1995)
 * marginal-conditional update and is exact for the DP: the truncated SBP
 * tail sticks are auxiliary variables that alpha does not need to see.
 *
 * @param k             number of OCCUPIED clusters (distinct labels with
 *                      >= 1 obs), 0 < k <= n.
 * @param n             number of observations, n >= 1.
 * @param a             Gamma prior shape (a_alpha), > 0.
 * @param b             Gamma prior rate  (b_alpha), > 0.
 * @param alpha_current current alpha; needed to form the auxiliary draw
 *                      eta ~ Beta(alpha_current + 1, n), > 0.
 * @param rng           RNG (std::mt19937_64).
 * @return              a fresh draw of alpha > 0.
 *
 * Numerical notes: eta is clamped away from 0 and 1 so log eta is finite
 * and the Gamma rate (b - log eta) stays strictly positive (log eta < 0).
 * When k == 0 (no occupied clusters -- should not occur with n >= 1) the
 * second mixture component's shape a + k - 1 may be <= 0; we then force
 * the first component, whose shape a + k = a > 0 is always valid.
 */
inline double sample_alpha_escobar_west(std::size_t k, std::size_t n,
                                        double a, double b,
                                        double alpha_current,
                                        std::mt19937_64& rng) {
    if (!(a > 0.0)) {
        throw std::invalid_argument(
            "bnp::sample_alpha_escobar_west: a (prior shape) must be > 0");
    }
    if (!(b > 0.0)) {
        throw std::invalid_argument(
            "bnp::sample_alpha_escobar_west: b (prior rate) must be > 0");
    }
    if (n == 0) {
        throw std::invalid_argument(
            "bnp::sample_alpha_escobar_west: n must be >= 1");
    }
    if (k > n) {
        throw std::invalid_argument(
            "bnp::sample_alpha_escobar_west: k must be <= n");
    }
    if (!(alpha_current > 0.0) || !std::isfinite(alpha_current)) {
        throw std::invalid_argument(
            "bnp::sample_alpha_escobar_west: alpha_current must be finite "
            "and > 0");
    }

    const double k_d = static_cast<double>(k);
    const double n_d = static_cast<double>(n);

    // ---- Step 1: eta | alpha, n ~ Beta(alpha + 1, n) ------------------
    // gamma-trick: X ~ Gamma(alpha+1, 1), Y ~ Gamma(n, 1), eta = X/(X+Y).
    double eta;
    {
        std::gamma_distribution<double> gam_x(alpha_current + 1.0, 1.0);
        std::gamma_distribution<double> gam_y(n_d, 1.0);
        double x = 0.0, y = 0.0, s = 0.0;
        for (int retry = 0; retry < 8; ++retry) {
            x = gam_x(rng);
            y = gam_y(rng);
            s = x + y;
            if (s > 0.0 && std::isfinite(s)) break;
        }
        if (!(s > 0.0) || !std::isfinite(s)) {
            // Degenerate auxiliary draw; fall back to a prior draw so the
            // chain never stalls. (Statistically negligible — only fires on
            // total gamma underflow, which needs alpha_current ~ 0.)
            std::gamma_distribution<double> gam_prior(a, 1.0 / b);
            return gam_prior(rng);
        }
        eta = x / s;
        // Clamp away from 0 and 1: log eta must be finite and < 0 so the
        // Gamma rate (b - log eta) stays strictly positive.
        if (eta < 1e-300)      eta = 1e-300;
        if (eta > 1.0 - 1e-15) eta = 1.0 - 1e-15;
    }

    const double log_eta = std::log(eta);          // < 0
    const double rate    = b - log_eta;            // > b > 0

    // ---- Step 2: alpha | eta, k -- 2-component Gamma mixture ----------
    // Escobar & West (1995) eq. (14): with mixing weight pi_eta defined by
    //
    //     pi_eta / (1 - pi_eta) = (a + k - 1) / ( n * (b - log eta) ),
    //
    //   with prob  pi_eta      : alpha ~ Gamma(a + k,     b - log eta)
    //   with prob  1 - pi_eta  : alpha ~ Gamma(a + k - 1, b - log eta).
    //
    // i.e. the ODDS above is the odds of the HIGH-shape (a + k) component,
    // so pi_eta = odds / (1 + odds) is the probability of Gamma(a + k).
    const double shape_hi = a + k_d;          // always > 0
    const double shape_lo = a + k_d - 1.0;    // may be <= 0 when k == 0

    double alpha_new;
    if (!(shape_lo > 0.0)) {
        // k == 0 edge case: only the high-shape component is valid.
        std::gamma_distribution<double> gam_hi(shape_hi, 1.0 / rate);
        alpha_new = gam_hi(rng);
    } else {
        // odds = pi_eta / (1 - pi_eta) = (a + k - 1) / (n * rate);
        // pi_hi = pi_eta = odds / (1 + odds) is the weight on Gamma(a + k).
        const double odds  = shape_lo / (n_d * rate);
        const double pi_hi = odds / (1.0 + odds);
        std::uniform_real_distribution<double> unif(0.0, 1.0);
        if (unif(rng) < pi_hi) {
            std::gamma_distribution<double> gam_hi(shape_hi, 1.0 / rate);
            alpha_new = gam_hi(rng);
        } else {
            std::gamma_distribution<double> gam_lo(shape_lo, 1.0 / rate);
            alpha_new = gam_lo(rng);
        }
    }

    if (!(alpha_new > 0.0) || !std::isfinite(alpha_new)) {
        // Guard against a degenerate gamma draw; resample from the prior.
        std::gamma_distribution<double> gam_prior(a, 1.0 / b);
        alpha_new = gam_prior(rng);
    }
    return alpha_new;
}

// ----------------------------------------------------------------------------
// CRP log-prior
// ----------------------------------------------------------------------------

/**
 * @brief Log-prior probability under a Chinese Restaurant Process that the
 *        next draw lands in cluster k (0-indexed, in {0, ..., K-1, K}).
 *
 *   p(k | n_minus_i, alpha, N_minus_i)  ∝
 *       n_minus_i[k]            for k < n_minus_i.n_elem (occupied)
 *       alpha                   for k == n_minus_i.n_elem (NEW cluster)
 *
 *   Normalising constant is (N_minus_i + alpha).
 *
 * @param k          0-indexed cluster id; k == n_minus_i.n_elem means NEW.
 * @param n_minus_i  cluster counts EXCLUDING the current observation.
 * @param alpha      DP concentration parameter, > 0.
 * @param N_minus_i  total count = arma::sum(n_minus_i); passed in to
 *                   avoid recomputation in inner loops.
 * @return log probability (normalised).
 */
inline double crp_log_prior(std::size_t k, const arma::vec& n_minus_i,
                            double alpha, std::size_t N_minus_i) {
    if (!(alpha > 0.0)) {
        throw std::invalid_argument("bnp::crp_log_prior: alpha must be > 0");
    }
    const double denom = static_cast<double>(N_minus_i) + alpha;
    if (k < n_minus_i.n_elem) {
        const double n_k = n_minus_i[k];
        if (!(n_k > 0.0)) {
            // empty cluster — treat as -inf (NEW cluster slot is k = K)
            return -std::numeric_limits<double>::infinity();
        }
        return std::log(n_k) - std::log(denom);
    } else if (k == n_minus_i.n_elem) {
        return std::log(alpha) - std::log(denom);
    }
    throw std::out_of_range(
        "bnp::crp_log_prior: k out of {0, ..., n_minus_i.n_elem}");
}

// ----------------------------------------------------------------------------
// Pitman-Yor log-prior
// ----------------------------------------------------------------------------

/**
 * @brief Log-prior under a Pitman-Yor process (Pitman & Yor 1997).
 *
 *   p(k | n_minus_i, K_current, alpha, discount, N_minus_i)  ∝
 *       (n_minus_i[k] - discount)     for k < K_current (occupied)
 *       alpha + K_current * discount   for k == K_current (NEW cluster)
 *
 *   When discount == 0 this reduces to the CRP. When discount > 0
 *   (typically in [0, 1)), the marginal distribution of the number of
 *   distinct clusters has a heavier tail than CRP (power-law).
 *
 * @param k           0-indexed cluster id; k == K_current means NEW.
 * @param n_minus_i   cluster counts EXCLUDING the current observation.
 * @param K_current   current number of POPULATED clusters; equals
 *                    number of strictly-positive entries in n_minus_i.
 *                    Passed explicitly because some samplers maintain it
 *                    independently for efficiency.
 * @param alpha       PY concentration, > -discount.
 * @param discount    PY discount parameter, in [0, 1).
 * @param N_minus_i   total count = arma::sum(n_minus_i).
 */
inline double py_log_prior(std::size_t k, const arma::vec& n_minus_i,
                           std::size_t K_current, double alpha,
                           double discount, std::size_t N_minus_i) {
    if (!(discount >= 0.0 && discount < 1.0)) {
        throw std::invalid_argument(
            "bnp::py_log_prior: discount must be in [0, 1)");
    }
    if (!(alpha > -discount)) {
        throw std::invalid_argument(
            "bnp::py_log_prior: alpha must be > -discount");
    }
    const double denom = static_cast<double>(N_minus_i) + alpha;
    if (k < K_current) {
        if (k >= n_minus_i.n_elem) {
            throw std::out_of_range("bnp::py_log_prior: k indexes outside counts");
        }
        const double n_k = n_minus_i[k];
        if (!(n_k > 0.0)) {
            return -std::numeric_limits<double>::infinity();
        }
        return std::log(n_k - discount) - std::log(denom);
    } else if (k == K_current) {
        return std::log(alpha + static_cast<double>(K_current) * discount)
             - std::log(denom);
    }
    throw std::out_of_range(
        "bnp::py_log_prior: k out of {0, ..., K_current}");
}

// ----------------------------------------------------------------------------
// CRP sampling (predict-time / Neal-Alg-2 helper)
// ----------------------------------------------------------------------------

/**
 * @brief Draw an assignment from the CRP predictive distribution.
 *
 *   Given cluster_counts (length K) and concentration alpha,
 *   sample a 0-indexed cluster id in {0, ..., K} where K means
 *   "create a new cluster".
 *
 * Use case: predict-time / posterior-predictive in user-written
 * stochastic refreshers. The `cluster_counts` here are the COMPLETE
 * counts (n_minus_i is not subtracted; this is for adding a fresh
 * observation that has no current label).
 *
 * @return 0-indexed id in [0, cluster_counts.n_elem]; the value
 *         cluster_counts.n_elem signals "NEW cluster".
 */
inline std::size_t crp_sample_new_assignment(
        const arma::vec& cluster_counts, double alpha,
        std::mt19937_64& rng) {
    if (!(alpha > 0.0)) {
        throw std::invalid_argument(
            "bnp::crp_sample_new_assignment: alpha must be > 0");
    }
    const std::size_t K = cluster_counts.n_elem;
    const double n_total = arma::sum(cluster_counts);
    const double denom = n_total + alpha;
    // Cumulative weights up to K, then alpha for the NEW slot.
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    const double u = uniform(rng) * denom;
    double cumul = 0.0;
    for (std::size_t k = 0; k < K; ++k) {
        cumul += cluster_counts[k];
        if (u < cumul) return k;
    }
    return K;  // NEW cluster
}

/**
 * @brief Pitman-Yor variant of @ref crp_sample_new_assignment.
 *
 *   Weights:
 *       (cluster_counts[k] - discount)  for k = 0, ..., K_current - 1
 *       alpha + K_current * discount     for k = K_current   (NEW)
 *
 * `K_current` is the count of POPULATED clusters; the function does
 * NOT scan cluster_counts to compute it (avoids ambiguity in case
 * the user maintains its own representation with explicit empties).
 */
inline std::size_t py_sample_new_assignment(
        const arma::vec& cluster_counts, double alpha, double discount,
        std::size_t K_current, std::mt19937_64& rng) {
    if (!(discount >= 0.0 && discount < 1.0)) {
        throw std::invalid_argument(
            "bnp::py_sample_new_assignment: discount must be in [0, 1)");
    }
    if (!(alpha > -discount)) {
        throw std::invalid_argument(
            "bnp::py_sample_new_assignment: alpha must be > -discount");
    }
    const double n_total = arma::sum(cluster_counts);
    const double new_w = alpha + static_cast<double>(K_current) * discount;
    const double denom = n_total + alpha;
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    const double u = uniform(rng) * denom;
    double cumul = 0.0;
    for (std::size_t k = 0; k < K_current; ++k) {
        const double w = cluster_counts[k] - discount;
        if (w > 0.0) {
            cumul += w;
            if (u < cumul) return k;
        }
    }
    if (u < cumul + new_w) return K_current;
    // Numerical fallback for tiny round-off — return NEW.
    return K_current;
}

}  // namespace bnp
}  // namespace AI4BayesCode

#endif  // AI4BAYESCODE_BNP_UTILS_HPP

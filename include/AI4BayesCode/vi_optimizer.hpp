/*================================================================================
 *  AI4BayesCode: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v2.0 or later
 *  (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
 *================================================================================
 *
 *  vi_optimizer.hpp  --  RAABBVI optimizer helpers (Welandawe 2022).
 *
 *  Header-only port of the Welandawe et al. 2022 "Robust, Automated,
 *  and Accurate Black-box Variational Inference" algorithm. This file
 *  exposes the UTILITIES that VI blocks compose:
 *
 *    - raabbvi_config            : POD struct of optimizer hyperparams
 *    - avgAdam_state             : per-block optimizer state
 *    - avgAdam_step              : one avgAdam update with Polyak-Ruppert
 *                                  iterate averaging
 *    - skl_mean_field_gaussian   : closed-form symmetric KL between two
 *                                  mean-field Gaussian variational q's
 *    - psis_khat                 : Pareto-smoothed importance sampling k̂
 *                                  (Vehtari, Simpson, Gelman, Yao, Gabry
 *                                  2024 — JMLR 25:72)
 *
 *  The ORCHESTRATION (within-γ inner loop + between-γ shrinkage + SKL
 *  termination) lives inside each concrete vi_block subclass that
 *  composes these utilities. v1 simplifies vs the paper:
 *
 *    - within-γ "convergence" = fixed inner_iter_per_epoch steps
 *      (no R̂-based dynamic detection)
 *    - SKL termination only — no RI (relative inefficiency) factor
 *    - final_khat is the joint k̂ over S samples from the converged q
 *
 *  v2 may add R̂-based dynamic inner convergence + RI. The simplification
 *  retains the practical-defaults essence (avgAdam, iterate averaging,
 *  geometric γ decay, SKL-based stopping).
 *
 *  See system_design.md §18.7 for the architectural backing.
 *================================================================================*/

#ifndef AI4BAYESCODE_VI_OPTIMIZER_HPP
#define AI4BAYESCODE_VI_OPTIMIZER_HPP

#include <algorithm>   // std::sort, std::max, std::min
#include <cmath>       // std::log, std::sqrt, std::exp, std::isfinite, std::lgamma
#include <cstddef>     // std::size_t
#include <limits>
#include <numeric>     // std::accumulate
#include <vector>

#ifndef MCMC_USE_RCPP_ARMADILLO
# include <armadillo>
#else
# include <RcppArmadillo.h>
#endif

namespace AI4BayesCode {
namespace vi_optimizer {

/**
 * @brief POD configuration for the RAABBVI-lite optimizer (v1).
 *
 * All defaults are taken from Welandawe 2022 §5 (experiments) modulo
 * the v1 simplifications noted in the file header.
 */
struct raabbvi_config {
    /// Initial learning rate for avgAdam at the first γ-epoch.
    double gamma_0 = 0.1;

    /// Geometric γ-decay factor between epochs: γ_{k+1} = ρ · γ_k.
    double rho = 0.5;

    /// SKL inefficiency threshold I < τ → terminate.
    double tau = 0.1;

    /// Fixed inner-loop iterations per γ-epoch (v1 simplification; v2
    /// will switch to R̂-based dynamic detection per Welandawe §3.2).
    std::size_t inner_iter_per_epoch = 200;

    /// Hard cap on outer epochs (γ shrinkages). Stops runaway runs.
    std::size_t max_epochs = 50;

    /// avgAdam numerical-stability constant (Adam's ε, added inside the
    /// sqrt denominator). Default matches PyTorch / TensorFlow Adam.
    double eps_stab = 1e-8;

    /// Number of q-samples for joint PSIS-k̂ at SKL termination.
    /// Dhaka 2021 calibrates this against the Pareto-fit variance.
    std::size_t S_khat = 1000;
};

// ---------------------------------------------------------------------------
//  avgAdam — squared-gradient cumulative average + Polyak-Ruppert
// ---------------------------------------------------------------------------

/**
 * @brief Per-block avgAdam optimizer state.
 *
 * Welandawe 2022 §3.1: the squared-gradient denominator is a cumulative
 * average (NOT an EMA as in plain Adam). This makes the update
 * asymptotically SGD-like and removes Adam's known late-stage drift
 * in VI. Iterate averaging (Polyak-Ruppert) on top gives the final
 * λ̄ estimate; raw λ is the iterate that gets the gradient applied
 * each step.
 *
 * Storage:
 *   - v_bar: cumulative-average squared gradient, dim D = 2K (μ + log σ)
 *            (or 2K + K(K-1)/2 = K(K+3)/2 for full-rank with packed L)
 *   - lambda_bar: Polyak-Ruppert iterate average, same dim as λ
 *   - t: total step count within the CURRENT γ-epoch (resets on shrink)
 *   - g_squared_sum: cumulative sum of g², used to compute v_bar
 *                    (we store the SUM and divide by t to get the
 *                     average — avoids drift from successive averaging)
 *   - lambda_sum: cumulative sum of λ (we store sum, divide by t for
 *                 lambda_bar, same numerical reason)
 *
 * Reset within-epoch: at the start of each new γ-epoch, t→0, g_squared_sum
 * → 0, lambda_sum → 0 — but v_bar and lambda_bar are kept around as the
 * end-of-epoch summary for SKL termination.
 */
struct avgAdam_state {
    arma::vec   v_bar;          ///< cumulative-avg squared grad (= g_squared_sum / t)
    arma::vec   lambda_bar;     ///< Polyak-Ruppert avg of λ (= lambda_sum / t)
    arma::vec   g_squared_sum;  ///< running sum of g² since epoch start
    arma::vec   lambda_sum;     ///< running sum of λ since epoch start
    std::size_t t = 0;          ///< step count within current epoch

    /// Reset for a new γ-epoch. v_bar and lambda_bar are passed in as
    /// the END-of-previous-epoch values (callers cache them externally
    /// for SKL termination computation).
    void reset_epoch(std::size_t D) {
        v_bar.zeros(D);
        lambda_bar.zeros(D);
        g_squared_sum.zeros(D);
        lambda_sum.zeros(D);
        t = 0;
    }
};

/**
 * @brief One avgAdam update step.
 *
 * Applied at step t+1 of the current γ-epoch. Inputs:
 *   - lambda  : current λ (in/out, will be updated in place)
 *   - state   : optimizer state (modified in place)
 *   - grad    : raw gradient g_{t+1} of -ELBO at the current λ
 *   - gamma   : learning rate for the current γ-epoch
 *   - cfg     : optimizer config (uses cfg.eps_stab)
 *
 * Update rule:
 *   t+1                = t + 1
 *   g_squared_sum     += g²    (element-wise)
 *   v_bar             = g_squared_sum / (t+1)
 *   λ_new             = λ - γ · g / sqrt(v_bar + ε_stab)
 *   lambda_sum        += λ_new
 *   lambda_bar        = lambda_sum / (t+1)
 *   λ                 = λ_new
 *
 * The cumulative-average denominator (vs EMA) is the avgAdam
 * distinction from plain Adam.
 */
inline void avgAdam_step(arma::vec&            lambda,
                         avgAdam_state&        state,
                         const arma::vec&      grad,
                         double                gamma,
                         const raabbvi_config& cfg) {
    state.t += 1;
    state.g_squared_sum += arma::square(grad);
    state.v_bar = state.g_squared_sum / static_cast<double>(state.t);

    // avgAdam step: divide gradient by sqrt(v_bar + eps_stab) element-wise.
    lambda -= gamma * (grad / arma::sqrt(state.v_bar + cfg.eps_stab));

    state.lambda_sum += lambda;
    state.lambda_bar = state.lambda_sum / static_cast<double>(state.t);
}

// ---------------------------------------------------------------------------
//  SKL termination — mean-field Gaussian closed form
// ---------------------------------------------------------------------------

/**
 * @brief Symmetric KL between two mean-field Gaussian variational
 *        distributions q_a = ∏_i N(μ_a,i, σ_a,i²) and q_b similarly.
 *
 * SKL(q_a, q_b) = KL(q_a || q_b) + KL(q_b || q_a)
 *
 * For independent Gaussians on R^K:
 *   KL(N(μ_a, σ_a²) || N(μ_b, σ_b²))
 *     = sum_i [ log(σ_b,i / σ_a,i)
 *              + (σ_a,i² + (μ_a,i - μ_b,i)²) / (2 σ_b,i²)
 *              - 1/2 ]
 *
 * Symmetric:
 *   SKL = sum_i [ (σ_a,i² + (μ_a,i - μ_b,i)²) / (2 σ_b,i²)
 *               + (σ_b,i² + (μ_b,i - μ_a,i)²) / (2 σ_a,i²)
 *               - 1 ]
 *   (the log-ratio terms cancel: log(σ_b/σ_a) + log(σ_a/σ_b) = 0)
 *
 * @param mu_a, log_sd_a   First q's mean / log σ (length K each)
 * @param mu_b, log_sd_b   Second q's mean / log σ (length K each)
 * @return non-negative SKL value
 */
inline double skl_mean_field_gaussian(const arma::vec& mu_a,
                                       const arma::vec& log_sd_a,
                                       const arma::vec& mu_b,
                                       const arma::vec& log_sd_b) {
    if (mu_a.n_elem != mu_b.n_elem ||
        log_sd_a.n_elem != log_sd_b.n_elem ||
        mu_a.n_elem != log_sd_a.n_elem) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const arma::vec sigma_a = arma::exp(log_sd_a);
    const arma::vec sigma_b = arma::exp(log_sd_b);
    const arma::vec sigma_a_sq = arma::square(sigma_a);
    const arma::vec sigma_b_sq = arma::square(sigma_b);
    const arma::vec d_mu_sq = arma::square(mu_a - mu_b);

    const arma::vec term1 = (sigma_a_sq + d_mu_sq) / (2.0 * sigma_b_sq);
    const arma::vec term2 = (sigma_b_sq + d_mu_sq) / (2.0 * sigma_a_sq);
    return arma::sum(term1 + term2 - 1.0);
}

// ---------------------------------------------------------------------------
//  PSIS-k̂ — Pareto-smoothed importance sampling shape parameter
// ---------------------------------------------------------------------------

namespace detail {

/**
 * Closed-form GPD shape estimator (Zhang & Stephens 2009 §3, "method 1"),
 * the variant used in PSIS (Vehtari et al. 2024 JMLR 25:72).
 *
 * Given M sorted positive samples y_1 <= ... <= y_M (ascending) from the
 * GPD upper tail, returns the bias-corrected k̂.
 *
 * Zhang-Stephens parameterization: GPD reparameterized with b = -k/sigma,
 * so MLE of k given b is
 *   k(b) = mean(log(1 - b*y))
 * Heavy-tail GPD (standard shape k_std > 0) corresponds to b < 0 in
 * this parameterization, giving k(b) > 0 (since y > 0, 1 - b*y > 1 when
 * b < 0, log > 0, mean > 0). Thin-tail (k_std < 0) gives b > 0,
 * k(b) < 0. The returned k̂ MATCHES the standard PSIS convention:
 *   k̂ > 0  : heavy-tailed importance weights (PSIS warning at >= 0.7)
 *   k̂ < 0  : thinner-than-exponential tail (q is at least as
 *             concentrated as p — best case for IS).
 *
 * Implementation follows arviz/stats/loo.py:gpdfit and R loo package.
 */
inline double gpd_fit_shape(const std::vector<double>& y) {
    const std::size_t M = y.size();
    if (M < 5) return std::numeric_limits<double>::quiet_NaN();

    const double y_M = y[M - 1];                              // max
    const std::size_t q25_idx = std::max<std::size_t>(0,
                                  static_cast<std::size_t>(0.25 * M + 0.5));
    const double y_q25 = y[std::min(q25_idx, M - 1)];
    if (!(y_M > 0.0) || !(y_q25 > 0.0)) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const double prior_factor = 3.0;
    const std::size_t mp_count = 30 + static_cast<std::size_t>(std::sqrt(M));

    // Candidate b grid (Zhang-Stephens 2009 prior).
    //   bs[j] = 1/y_M + (1 - sqrt(mp_count / (j+0.5))) / (prior * y_q25)
    // For j << mp_count, the sqrt factor is large → bs[j] strongly
    // negative. For j == mp_count - 1, the factor is ≈ 1 → bs near 1/y_M.
    // The grid spans the range of plausible b including the typical
    // negative-b heavy-tail region.
    std::vector<double> bs(mp_count);
    for (std::size_t i = 0; i < mp_count; ++i) {
        const double j = static_cast<double>(i + 1);
        const double term = 1.0 - std::sqrt(static_cast<double>(mp_count) /
                                            (j - 0.5));
        bs[i] = 1.0 / y_M + term / (prior_factor * y_q25);
    }

    // Profile log-likelihood at each b (Zhang-Stephens eq. derived from
    // GPD log-likelihood):
    //   l(b) = M * (log(-b/k) - k - 1)   where k = mean(log(1 - b*y))
    auto profile_ll = [&y, M](double b) -> double {
        double sum_log = 0.0;
        for (double yi : y) {
            const double arg = 1.0 - b * yi;
            if (arg <= 0.0) return -std::numeric_limits<double>::infinity();
            sum_log += std::log(arg);
        }
        const double k = sum_log / static_cast<double>(M);   // NOT negated
        if (b == 0.0 || k == 0.0 || -b / k <= 0.0) {
            return -std::numeric_limits<double>::infinity();
        }
        return static_cast<double>(M) * (std::log(-b / k) - k - 1.0);
    };

    std::vector<double> ll(mp_count);
    double max_ll = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < mp_count; ++i) {
        ll[i] = profile_ll(bs[i]);
        if (std::isfinite(ll[i]) && ll[i] > max_ll) max_ll = ll[i];
    }
    if (!std::isfinite(max_ll)) return std::numeric_limits<double>::quiet_NaN();

    // Bayesian-averaged b (Zhang-Stephens posterior).
    double w_sum = 0.0;
    double b_w_sum = 0.0;
    for (std::size_t i = 0; i < mp_count; ++i) {
        const double w = std::exp(ll[i] - max_ll);
        if (std::isfinite(w)) {
            w_sum += w;
            b_w_sum += w * bs[i];
        }
    }
    if (!(w_sum > 0.0)) return std::numeric_limits<double>::quiet_NaN();
    const double b_hat = b_w_sum / w_sum;

    // k̂ at b_hat (no sign flip).
    double sum_log = 0.0;
    for (double yi : y) {
        const double arg = 1.0 - b_hat * yi;
        if (arg <= 0.0) return std::numeric_limits<double>::quiet_NaN();
        sum_log += std::log(arg);
    }
    const double k = sum_log / static_cast<double>(M);

    // Small-sample bias correction (loo package convention):
    //   k_hat ← (M * k + 5) / (M + 10)
    return (k * static_cast<double>(M) + 10.0 * 0.5) /
           (static_cast<double>(M) + 10.0);
}

} // namespace detail

/**
 * @brief Pareto-Smoothed Importance Sampling shape parameter k̂.
 *
 * Yao et al. 2018, Vehtari et al. 2024 JMLR 25:72. Used as the
 * Layer-3 R2-VI diagnostic for VI blocks (Check #23 in validator.md).
 *
 * Threshold interpretation (Yao 2018 + Dhaka 2021):
 *   k̂ < 0.5         : PASS — q is a good approximation; expectations
 *                      via plain Monte Carlo are reliable.
 *   0.5 ≤ k̂ < 0.7   : CAUTION — Monte Carlo expectations need PSIS
 *                      reweighting; CIs may be biased small.
 *   k̂ ≥ 0.7         : FAIL — q is a poor approximation; PSIS cannot
 *                      rescue. Run the action chain from
 *                      validator.md §R2-VI before reporting.
 *
 * Algorithm:
 *   1. Convert log_weights → linear weights, subtracting max for
 *      numerical stability.
 *   2. Sort descending; take the top M = ceil(min(S/5, 3*sqrt(S)))
 *      samples (Vehtari et al. PSIS default).
 *   3. Subtract the (M+1)-th sample value (the tail threshold) from
 *      the top M to get the GPD-distributed excesses.
 *   4. Fit GPD via Zhang-Stephens 2009 closed-form Bayesian estimator
 *      with small-sample bias correction.
 *
 * Returns NaN if S < 25 (insufficient samples for a meaningful k̂).
 *
 * @param log_weights  S log importance weights log w_s = log p̃(η_s) - log q(η_s)
 * @return joint k̂ estimate, or NaN if S too small / degenerate.
 */
inline double psis_khat(const arma::vec& log_weights) {
    const std::size_t S = log_weights.n_elem;
    if (S < 25) return std::numeric_limits<double>::quiet_NaN();

    // Convert to linear weights (subtract max for numerical stability).
    const double max_lw = log_weights.max();
    if (!std::isfinite(max_lw)) return std::numeric_limits<double>::quiet_NaN();

    // Spread check: if log-weights are essentially constant, q is a
    // near-perfect approximation and the GPD upper tail is degenerate
    // (all excesses ~ 0). Returning NaN would mislead the Layer-3
    // verdict; return a clearly negative sentinel to indicate
    // "k̂ far below the PASS threshold". This is the PSIS convention
    // (Vehtari et al. 2024 §4): k̂ < 0 means "thinner-than-exponential
    // tail" — i.e., q is at least as concentrated as p.
    const double min_lw = log_weights.min();
    if ((max_lw - min_lw) < 1e-10) {
        return -1.0;
    }

    std::vector<double> w(S);
    for (std::size_t i = 0; i < S; ++i) {
        w[i] = std::exp(log_weights[i] - max_lw);
    }

    // Sort ascending (we want the upper tail).
    std::sort(w.begin(), w.end());

    // M = ceil(min(S/5, 3*sqrt(S))) — Vehtari PSIS default tail size.
    const std::size_t M = std::min(static_cast<std::size_t>(
                                       std::ceil(3.0 * std::sqrt(
                                           static_cast<double>(S)))),
                                    S / 5);
    if (M < 5) return std::numeric_limits<double>::quiet_NaN();

    // Tail starts at index (S - M); threshold = the (S - M - 1)-th value.
    if (S < M + 1) return std::numeric_limits<double>::quiet_NaN();
    const double threshold = w[S - M - 1];

    // Excesses y_i = w_i - threshold, for i in tail.
    std::vector<double> y;
    y.reserve(M);
    for (std::size_t i = S - M; i < S; ++i) {
        const double yi = w[i] - threshold;
        if (yi > 0.0) y.push_back(yi);
    }
    if (y.size() < 5) return std::numeric_limits<double>::quiet_NaN();

    // Tail-degeneracy guard: if even the top excess is essentially 0
    // (numerical near-tie with the threshold), report "q is much
    // better than the PSIS detection limit" — same negative sentinel
    // as the constant-weight case above.
    if (y.back() < 1e-12) {
        return -1.0;
    }

    // y is already sorted (since w was sorted and we shifted by a constant).
    return detail::gpd_fit_shape(y);
}

// ---------------------------------------------------------------------------
//  SKL termination check
// ---------------------------------------------------------------------------

/**
 * @brief Convenience: check whether SKL between successive epoch
 *        averages indicates termination.
 *
 * v1 uses RSKL only (skipping RI from Welandawe 2022 §3.4):
 *   RSKL_k = SKL(λ̄_k, λ̄_{k-1}) / SKL(λ̄_k, λ̄_0)
 *   terminate iff RSKL_k < τ
 *
 * SKL_k0 (denominator) is the SKL relative to the initial estimate
 * λ̄_0. This normalizes so τ has the same meaning across problems with
 * different absolute SKL scales. If SKL_k0 is ~0 (degenerate init or
 * everything converged at epoch 0), we cannot evaluate the ratio and
 * conservatively return false (keep going).
 *
 * @param skl_consecutive  SKL(λ̄_k, λ̄_{k-1})
 * @param skl_initial      SKL(λ̄_k, λ̄_0)
 * @param tau              user threshold (default 0.1)
 * @return true if RSKL < tau (terminate)
 */
inline bool skl_terminate(double skl_consecutive,
                           double skl_initial,
                           double tau) {
    if (!(skl_initial > 0.0) || !std::isfinite(skl_consecutive)) {
        return false;
    }
    return (skl_consecutive / skl_initial) < tau;
}

} // namespace vi_optimizer
} // namespace AI4BayesCode

#endif // AI4BAYESCODE_VI_OPTIMIZER_HPP

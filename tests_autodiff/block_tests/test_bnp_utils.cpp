// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
// test_bnp_utils.cpp
//
// LIBRARY-LEVEL parity test for the free functions in
// include/AI4BayesCode/bnp_utils.hpp (namespace AI4BayesCode::bnp):
//
//     counts_from_z              crp_log_prior            py_log_prior
//     crp_sample_new_assignment  py_sample_new_assignment
//     sample_alpha_escobar_west
//
// bnp_utils.hpp is not a block, so there is no step() to iterate. What is
// verified here is each function's MECHANISM against a target stated in
// closed form, independent of any particular mixture model that calls it.
// Every example that composes a DP / PY mixture (DPGaussianMixture,
// PYGaussianMixture, FiniteGaussianMixture, DPGaussianMixture_DerivedAlpha)
// inherits correctness of these primitives from this test.
//
// Three different kinds of target are used, one per function class, because
// the functions differ in kind.
//
// PART A -- deterministic functions: EXACT identity, round-off tolerance
// ---------------------------------------------------------------------
// counts_from_z, crp_log_prior and py_log_prior are pure functions with a
// closed-form value, so there is nothing statistical to check: the test
// compares against hand-computed numbers and against two exact identities.
//
//   A1. counts_from_z reproduces a hand-written histogram exactly, and
//       THROWS on an out-of-range label and on K == 0 (silently dropping a
//       label would corrupt the sufficient statistic, so the throw is part
//       of the contract).
//   A2. crp_log_prior is normalised: with counts (3, 1, 6) and alpha = 2,
//       sum over k in {0,1,2,NEW} of exp(crp_log_prior) == 1, and the
//       individual values equal log(n_k / (N + alpha)) and
//       log(alpha / (N + alpha)).
//   A3. py_log_prior is normalised for discount d in (0, 1): the occupied
//       weights (n_k - d) sum to N - K d and the new-table weight is
//       alpha + K d, so the total is again N + alpha; and py_log_prior with
//       d == 0 reduces to crp_log_prior EXACTLY (same normaliser, same
//       weights). Tolerance 1e-12 -- these are sums of at most a dozen
//       double operations, whose accumulated round-off is O(1e-15).
//
// PART B -- finite-support samplers: multinomial frequencies vs the pmf
// --------------------------------------------------------------------
// crp_sample_new_assignment and py_sample_new_assignment draw ONE label
// from a finite pmf that is itself closed form (the CRP / PY predictive
// weights normalised by N + alpha), and successive calls are i.i.d. -- no
// Markov chain is involved. The target is therefore the pmf itself, and
// the check is a frequency comparison. Tolerance per category is
// 4 * sqrt(p (1 - p) / N) with N = 200000: the exact binomial standard
// error of the empirical frequency, at 4 SE (two-sided false-alarm rate
// about 6e-5 per category).
//
// PART C -- sample_alpha_escobar_west: ergodic averages vs the exact
//           Antoniak marginal posterior, obtained by quadrature
// ------------------------------------------------------------------
// This one has no closed-form one-step marginal: it is a two-step
// auxiliary-variable Gibbs kernel on the pair (alpha, eta),
//
//     eta   | alpha, n  ~  Beta(alpha + 1, n)
//     alpha | eta, k    ~  w Gamma(a + k, b - log eta)
//                        + (1 - w) Gamma(a + k - 1, b - log eta),
//
// so a single draw is not distributed as anything nameable. Exact
// enumeration does not apply either -- the state space is continuous.
// What IS known in closed form (up to a constant that does not involve
// alpha) is the distribution the kernel must leave INVARIANT. Antoniak
// (1974) gives p(k | alpha, n) = |s(n,k)| alpha^k Gamma(alpha)/Gamma(alpha+n),
// so under a Gamma(a, b) [shape-rate] prior the marginal posterior is
//
//     p(alpha | k, n)  proportional to
//         alpha^(a + k - 1) exp(-b alpha) Gamma(alpha) / Gamma(alpha + n),
//
// i.e. log p = (a + k - 1) log alpha - b alpha
//             + lgamma(alpha) - lgamma(alpha + n)  + const.
//
// That is a smooth, rapidly decaying 1-D density, so it is normalised here
// by composite trapezoid quadrature on a uniform grid over (0, A_MAX] with
// step h = 1e-4, from which the exact mean, variance, fourth central
// moment and quartiles are read off. Quadrature error at this step is
// O(h^2) relative, about 1e-9 -- five orders of magnitude below the Monte
// Carlo tolerances below, so the quadrature values are treated as exact.
// The test then runs the kernel as a chain and compares its ergodic
// averages and empirical CDF to those quadrature values. This is the
// invariance check: an implementation that (for example) put the mixing
// weight on the wrong Gamma component still produces a perfectly
// well-behaved chain, but converges to the WRONG alpha distribution, and
// this comparison sees it (that particular error moves the posterior mean
// by roughly 15 percent, more than ten times the tolerance).
//
// Two (k, n) configurations are used so the check covers both a small-alpha
// and a larger-alpha regime:
//     C1: k = 5,  n = 50,  a = 2, b = 4
//     C2: k = 20, n = 100, a = 1, b = 1
//
// Monte Carlo tolerances in Part C. The draws are serially correlated, so
// an i.i.d. standard error would be optimistic. Every Part-C threshold is
// an i.i.d. standard error inflated by sqrt(IACT_CAP) with IACT_CAP = 10,
// i.e. it assumes the integrated autocorrelation time is at most 10, and
// then takes 4 such standard errors:
//     mean      tol = 4 sqrt(IACT_CAP Var / N)
//     variance  tol = 4 sqrt(IACT_CAP (mu4 - Var^2) / N)   [delta method]
//     CDF at q  tol = 4 sqrt(IACT_CAP p (1 - p) / N)
// The IACT_CAP assumption is not asserted, it is CHECKED: a batch-means
// standard error of the mean (200 batches of 1000) is computed and the
// test FAILS unless it stays below sqrt(IACT_CAP Var / N). So a tolerance
// can never be silently widened by bad mixing -- bad mixing fails on its
// own flag.
//
// Returned Rcpp::List so the caller (tests_autodiff/run_all_parity.R) can
// check all_pass programmatically.
// ============================================================================

// [[Rcpp::depends(RcppArmadillo)]]

#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif

#include <RcppArmadillo.h>

#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/bnp_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

// ----------------------------------------------------------------------------
// Exact summaries of p(alpha | k, n) by quadrature (Part C reference).
// ----------------------------------------------------------------------------

struct alpha_exact {
    double mean;     // E[alpha | k, n]
    double var;      // Var[alpha | k, n]
    double mu4;      // E[(alpha - mean)^4 | k, n]
    double q[3];     // grid points nearest the 0.25 / 0.50 / 0.75 quantiles
    double p_at_q[3];// exact P(alpha <= q[j] | k, n) at those grid points
};

// log of the unnormalised Antoniak marginal posterior of alpha:
//   (a + k - 1) log alpha - b alpha + lgamma(alpha) - lgamma(alpha + n).
inline double log_alpha_post_unnorm(double alpha, double k_d, double n_d,
                                    double a, double b) {
    return (a + k_d - 1.0) * std::log(alpha) - b * alpha
         + std::lgamma(alpha) - std::lgamma(alpha + n_d);
}

// Composite trapezoid on the uniform grid alpha_j = j h, j = 1..M,
// h = A_max / M. The integrand vanishes at alpha = 0 (the exponent
// a + k - 1 exceeds 1 in both configurations, and lgamma(alpha) only
// diverges logarithmically) and is below 1e-100 relative at A_max, so the
// endpoint half-weights are dropped; the common factor h cancels from every
// ratio computed below.
alpha_exact alpha_posterior_exact(std::size_t k, std::size_t n,
                                  double a, double b,
                                  double A_max, std::size_t M) {
    const double k_d = static_cast<double>(k);
    const double n_d = static_cast<double>(n);
    const double h   = A_max / static_cast<double>(M);

    std::vector<double> f(M);
    double log_max = -std::numeric_limits<double>::infinity();
    for (std::size_t j = 1; j <= M; ++j) {
        const double alpha = static_cast<double>(j) * h;
        const double lf = log_alpha_post_unnorm(alpha, k_d, n_d, a, b);
        f[j - 1] = lf;
        if (lf > log_max) log_max = lf;
    }
    double S0 = 0.0, S1 = 0.0, S2 = 0.0, S3 = 0.0, S4 = 0.0;
    for (std::size_t j = 1; j <= M; ++j) {
        const double alpha = static_cast<double>(j) * h;
        const double w = std::exp(f[j - 1] - log_max);
        f[j - 1] = w;                      // reuse the buffer for the CDF
        S0 += w;
        S1 += w * alpha;
        S2 += w * alpha * alpha;
        S3 += w * alpha * alpha * alpha;
        S4 += w * alpha * alpha * alpha * alpha;
    }

    alpha_exact out;
    const double m1 = S1 / S0;
    const double m2 = S2 / S0;
    const double m3 = S3 / S0;
    const double m4 = S4 / S0;
    out.mean = m1;
    out.var  = m2 - m1 * m1;
    out.mu4  = m4 - 4.0 * m1 * m3 + 6.0 * m1 * m1 * m2 - 3.0 * m1 * m1 * m1 * m1;

    // Quartiles: walk the normalised cumulative sum. The reported target
    // probability is the cumulative value AT the chosen grid point, not the
    // nominal 0.25 / 0.50 / 0.75, so the grid resolution contributes no bias.
    const double targets[3] = {0.25, 0.50, 0.75};
    std::size_t t = 0;
    double cum = 0.0;
    for (std::size_t j = 1; j <= M && t < 3; ++j) {
        cum += f[j - 1];
        while (t < 3 && cum / S0 >= targets[t]) {
            out.q[t]      = static_cast<double>(j) * h;
            out.p_at_q[t] = cum / S0;
            ++t;
        }
    }
    for (; t < 3; ++t) { out.q[t] = A_max; out.p_at_q[t] = 1.0; }
    return out;
}

// ----------------------------------------------------------------------------
// One Part-C configuration: run the Escobar-West kernel and compare.
// ----------------------------------------------------------------------------

struct alpha_result {
    bool   pass;
    bool   pass_mean, pass_var, pass_mixing;
    bool   pass_cdf[3];
    double mean, var;
    double exact_mean, exact_var;
    double mcse_batch, mcse_cap;
    double tol_mean, tol_var;
    double emp_cdf[3], exact_cdf[3], tol_cdf[3];
};

alpha_result run_alpha_config(std::size_t k, std::size_t n, double a, double b,
                              std::size_t N_KEEP, std::size_t N_BURN,
                              std::uint64_t seed) {
    // IACT_CAP: every Monte Carlo tolerance below is an i.i.d. standard error
    // inflated by sqrt(IACT_CAP). The assumption is verified by the
    // batch-means check (pass_mixing) rather than assumed.
    const double IACT_CAP = 10.0;
    const double Nd = static_cast<double>(N_KEEP);

    const alpha_exact ex = alpha_posterior_exact(k, n, a, b,
                                                 /*A_max=*/80.0,
                                                 /*M=*/800000);

    std::mt19937_64 rng(seed);
    double alpha = a / b;                       // prior mean as the start
    for (std::size_t i = 0; i < N_BURN; ++i) {
        alpha = AI4BayesCode::bnp::sample_alpha_escobar_west(k, n, a, b,
                                                             alpha, rng);
    }
    std::vector<double> draws;
    draws.reserve(N_KEEP);
    for (std::size_t i = 0; i < N_KEEP; ++i) {
        alpha = AI4BayesCode::bnp::sample_alpha_escobar_west(k, n, a, b,
                                                             alpha, rng);
        draws.push_back(alpha);
    }

    double mean = 0.0;
    for (double v : draws) mean += v;
    mean /= Nd;
    double var = 0.0;
    for (double v : draws) { const double d = v - mean; var += d * d; }
    var /= (Nd - 1.0);

    // Batch means: 200 batches. sd(batch means) / sqrt(n_batches) estimates
    // the Monte Carlo standard error of `mean` including autocorrelation.
    const std::size_t n_batch = 200;
    const std::size_t bs = N_KEEP / n_batch;
    double bm_mean = 0.0;
    std::vector<double> bm(n_batch, 0.0);
    for (std::size_t bidx = 0; bidx < n_batch; ++bidx) {
        double s = 0.0;
        for (std::size_t i = 0; i < bs; ++i) s += draws[bidx * bs + i];
        bm[bidx] = s / static_cast<double>(bs);
        bm_mean += bm[bidx];
    }
    bm_mean /= static_cast<double>(n_batch);
    double bm_var = 0.0;
    for (double v : bm) { const double d = v - bm_mean; bm_var += d * d; }
    bm_var /= static_cast<double>(n_batch - 1);
    const double mcse_batch = std::sqrt(bm_var / static_cast<double>(n_batch));

    alpha_result r;
    r.mean = mean;
    r.var  = var;
    r.exact_mean = ex.mean;
    r.exact_var  = ex.var;

    // Tolerances (all absolute, all derived from the quadrature moments).
    r.tol_mean = 4.0 * std::sqrt(IACT_CAP * ex.var / Nd);
    r.tol_var  = 4.0 * std::sqrt(IACT_CAP * (ex.mu4 - ex.var * ex.var) / Nd);
    r.mcse_cap = std::sqrt(IACT_CAP * ex.var / Nd);

    r.pass_mean   = (std::abs(mean - ex.mean) < r.tol_mean);
    r.pass_var    = (std::abs(var  - ex.var)  < r.tol_var);
    r.pass_mixing = (mcse_batch < r.mcse_cap);
    r.mcse_batch  = mcse_batch;

    for (int j = 0; j < 3; ++j) {
        std::size_t below = 0;
        for (double v : draws) if (v <= ex.q[j]) ++below;
        const double p = ex.p_at_q[j];
        r.emp_cdf[j]   = static_cast<double>(below) / Nd;
        r.exact_cdf[j] = p;
        r.tol_cdf[j]   = 4.0 * std::sqrt(IACT_CAP * p * (1.0 - p) / Nd);
        r.pass_cdf[j]  = (std::abs(r.emp_cdf[j] - p) < r.tol_cdf[j]);
    }

    r.pass = r.pass_mean && r.pass_var && r.pass_mixing
          && r.pass_cdf[0] && r.pass_cdf[1] && r.pass_cdf[2];
    return r;
}

Rcpp::List alpha_result_to_list(const alpha_result& r) {
    return Rcpp::List::create(
        Rcpp::Named("pass")        = r.pass,
        Rcpp::Named("pass_mean")   = r.pass_mean,
        Rcpp::Named("pass_var")    = r.pass_var,
        Rcpp::Named("pass_mixing") = r.pass_mixing,
        Rcpp::Named("pass_cdf")    = Rcpp::LogicalVector::create(
            r.pass_cdf[0], r.pass_cdf[1], r.pass_cdf[2]),
        Rcpp::Named("mean")        = r.mean,
        Rcpp::Named("exact_mean")  = r.exact_mean,
        Rcpp::Named("tol_mean")    = r.tol_mean,
        Rcpp::Named("var")         = r.var,
        Rcpp::Named("exact_var")   = r.exact_var,
        Rcpp::Named("tol_var")     = r.tol_var,
        Rcpp::Named("mcse_batch")  = r.mcse_batch,
        Rcpp::Named("mcse_cap")    = r.mcse_cap,
        Rcpp::Named("emp_cdf")     = Rcpp::NumericVector::create(
            r.emp_cdf[0], r.emp_cdf[1], r.emp_cdf[2]),
        Rcpp::Named("exact_cdf")   = Rcpp::NumericVector::create(
            r.exact_cdf[0], r.exact_cdf[1], r.exact_cdf[2]),
        Rcpp::Named("tol_cdf")     = Rcpp::NumericVector::create(
            r.tol_cdf[0], r.tol_cdf[1], r.tol_cdf[2]));
}

}  // namespace

// [[Rcpp::export]]
Rcpp::List test_bnp_utils() {
    using namespace AI4BayesCode::bnp;

    // Round-off tolerance for the exact-identity checks in Part A. Each
    // value is a handful of double operations (a log, a division, an
    // exp, a sum of four terms), so the accumulated error is O(1e-15);
    // 1e-12 leaves three orders of magnitude of headroom and is still
    // far too tight to hide a formula error.
    const double TOL_EXACT = 1e-12;

    // ======================================================================
    // PART A1 -- counts_from_z: exact histogram + contract on bad input
    // ======================================================================
    bool a1_hist = true, a1_throw_range = false, a1_throw_K0 = false;
    {
        // labels are 1-indexed: 1,3,3,2,3,1 over K = 4
        // -> counts = (2, 1, 3, 0)
        const arma::vec z = arma::vec({1.0, 3.0, 3.0, 2.0, 3.0, 1.0});
        const arma::vec c = counts_from_z(z, 4);
        const arma::vec expect = arma::vec({2.0, 1.0, 3.0, 0.0});
        if (c.n_elem != 4) a1_hist = false;
        else for (std::size_t i = 0; i < 4; ++i)
            if (std::abs(c[i] - expect[i]) > 0.0) a1_hist = false;

        // A label of 0 is outside {1, ..., K}: must throw, not be dropped.
        try {
            (void) counts_from_z(arma::vec({1.0, 0.0}), 4);
        } catch (const std::exception&) { a1_throw_range = true; }
        // K == 0 is not a valid truncation level.
        try {
            (void) counts_from_z(arma::vec({1.0}), 0);
        } catch (const std::exception&) { a1_throw_K0 = true; }
    }
    const bool passA1 = a1_hist && a1_throw_range && a1_throw_K0;

    // ======================================================================
    // PART A2 -- crp_log_prior: hand values + exact normalisation
    // ======================================================================
    // counts (3, 1, 6), alpha = 2, N = 10, denominator N + alpha = 12.
    //   p(0) = 3/12   p(1) = 1/12   p(2) = 6/12   p(NEW) = 2/12
    bool a2_values = true, a2_norm = false, a2_empty_neginf = false;
    double crp_sum = 0.0;
    {
        const arma::vec nm = arma::vec({3.0, 1.0, 6.0});
        const double alpha = 2.0;
        const std::size_t N = 10;
        const double denom = 12.0;
        const double want[4] = {3.0 / denom, 1.0 / denom,
                                6.0 / denom, alpha / denom};
        for (std::size_t kk = 0; kk <= 3; ++kk) {
            const double lp = crp_log_prior(kk, nm, alpha, N);
            const double p  = std::exp(lp);
            crp_sum += p;
            if (std::abs(p - want[kk]) > TOL_EXACT) a2_values = false;
        }
        a2_norm = (std::abs(crp_sum - 1.0) < TOL_EXACT);

        // An occupied-slot entry with zero count is not a legal destination
        // (the NEW table has its own index), so its log-prior is -Inf.
        const arma::vec nm0 = arma::vec({3.0, 0.0, 6.0});
        const double lp0 = crp_log_prior(1, nm0, alpha, 9);
        a2_empty_neginf = std::isinf(lp0) && (lp0 < 0.0);
    }
    const bool passA2 = a2_values && a2_norm && a2_empty_neginf;

    // ======================================================================
    // PART A3 -- py_log_prior: exact normalisation, and d == 0 -> CRP
    // ======================================================================
    // With counts (5, 2, 1), K = 3, alpha = 1.5, d = 0.4:
    //   occupied weights 4.6, 1.6, 0.6  (sum 6.8 = N - K d = 8 - 1.2)
    //   new-table weight 1.5 + 3 * 0.4 = 2.7
    //   total 9.5 = N + alpha, so the four probabilities sum to 1 exactly.
    bool a3_values = true, a3_norm = false, a3_reduces_to_crp = true;
    double py_sum = 0.0;
    {
        const arma::vec nm = arma::vec({5.0, 2.0, 1.0});
        const double alpha = 1.5, d = 0.4;
        const std::size_t K = 3, N = 8;
        const double denom = 9.5;
        const double want[4] = {4.6 / denom, 1.6 / denom,
                                0.6 / denom, 2.7 / denom};
        for (std::size_t kk = 0; kk <= K; ++kk) {
            const double p = std::exp(py_log_prior(kk, nm, K, alpha, d, N));
            py_sum += p;
            if (std::abs(p - want[kk]) > TOL_EXACT) a3_values = false;
        }
        a3_norm = (std::abs(py_sum - 1.0) < TOL_EXACT);

        // Structural identity: Pitman-Yor with discount 0 IS the CRP.
        const double alpha_c = 2.0;
        const arma::vec nc = arma::vec({3.0, 1.0, 6.0});
        for (std::size_t kk = 0; kk <= 3; ++kk) {
            const double lp_py  = py_log_prior(kk, nc, 3, alpha_c, 0.0, 10);
            const double lp_crp = crp_log_prior(kk, nc, alpha_c, 10);
            if (std::abs(lp_py - lp_crp) > TOL_EXACT) a3_reduces_to_crp = false;
        }
    }
    const bool passA3 = a3_values && a3_norm && a3_reduces_to_crp;

    // ======================================================================
    // PART B -- finite-support samplers vs their pmf
    // ======================================================================
    // Successive calls are i.i.d. draws, so the tolerance is the plain
    // binomial standard error of a frequency, at 4 SE.
    const std::size_t N_CAT = 200000;
    const double Ncat_d = static_cast<double>(N_CAT);

    bool passB_crp = true;
    Rcpp::NumericVector crp_emp(4), crp_exact(4), crp_tol(4);
    {
        const arma::vec counts = arma::vec({3.0, 1.0, 6.0});
        const double alpha = 2.0, denom = 12.0;
        const double p[4] = {3.0 / denom, 1.0 / denom,
                             6.0 / denom, alpha / denom};
        std::vector<std::size_t> hit(4, 0);
        std::mt19937_64 rng(20260818ULL);
        for (std::size_t i = 0; i < N_CAT; ++i) {
            const std::size_t kk = crp_sample_new_assignment(counts, alpha, rng);
            if (kk > 3) { passB_crp = false; break; }
            ++hit[kk];
        }
        for (int j = 0; j < 4; ++j) {
            const double f = static_cast<double>(hit[j]) / Ncat_d;
            const double se = std::sqrt(p[j] * (1.0 - p[j]) / Ncat_d);
            crp_emp[j] = f; crp_exact[j] = p[j]; crp_tol[j] = 4.0 * se;
            if (std::abs(f - p[j]) >= 4.0 * se) passB_crp = false;
        }
    }

    bool passB_py = true;
    Rcpp::NumericVector py_emp(4), py_exact(4), py_tol(4);
    {
        const arma::vec counts = arma::vec({5.0, 2.0, 1.0});
        const double alpha = 1.5, d = 0.4, denom = 9.5;
        const std::size_t K = 3;
        const double p[4] = {4.6 / denom, 1.6 / denom,
                             0.6 / denom, 2.7 / denom};
        std::vector<std::size_t> hit(4, 0);
        std::mt19937_64 rng(20260819ULL);
        for (std::size_t i = 0; i < N_CAT; ++i) {
            const std::size_t kk =
                py_sample_new_assignment(counts, alpha, d, K, rng);
            if (kk > 3) { passB_py = false; break; }
            ++hit[kk];
        }
        for (int j = 0; j < 4; ++j) {
            const double f = static_cast<double>(hit[j]) / Ncat_d;
            const double se = std::sqrt(p[j] * (1.0 - p[j]) / Ncat_d);
            py_emp[j] = f; py_exact[j] = p[j]; py_tol[j] = 4.0 * se;
            if (std::abs(f - p[j]) >= 4.0 * se) passB_py = false;
        }
    }

    // ======================================================================
    // PART C -- Escobar-West alpha kernel vs the exact Antoniak posterior
    // ======================================================================
    const alpha_result c1 = run_alpha_config(/*k=*/5,  /*n=*/50,
                                             /*a=*/2.0, /*b=*/4.0,
                                             /*N_KEEP=*/200000,
                                             /*N_BURN=*/2000,
                                             /*seed=*/20260820ULL);
    const alpha_result c2 = run_alpha_config(/*k=*/20, /*n=*/100,
                                             /*a=*/1.0, /*b=*/1.0,
                                             /*N_KEEP=*/200000,
                                             /*N_BURN=*/2000,
                                             /*seed=*/20260821ULL);

    const bool all_pass = passA1 && passA2 && passA3
                       && passB_crp && passB_py
                       && c1.pass && c2.pass;

    return Rcpp::List::create(
        Rcpp::Named("all_pass")            = all_pass,
        Rcpp::Named("passA1_counts_from_z")= passA1,
        Rcpp::Named("passA2_crp_log_prior")= passA2,
        Rcpp::Named("passA3_py_log_prior") = passA3,
        Rcpp::Named("passB_crp_sampler")   = passB_crp,
        Rcpp::Named("passB_py_sampler")    = passB_py,
        Rcpp::Named("passC1_escobar_west") = c1.pass,
        Rcpp::Named("passC2_escobar_west") = c2.pass,
        Rcpp::Named("A_detail") = Rcpp::List::create(
            Rcpp::Named("hist_exact")       = a1_hist,
            Rcpp::Named("throws_bad_label") = a1_throw_range,
            Rcpp::Named("throws_K0")        = a1_throw_K0,
            Rcpp::Named("crp_values")       = a2_values,
            Rcpp::Named("crp_sum")          = crp_sum,
            Rcpp::Named("crp_empty_neg_inf")= a2_empty_neginf,
            Rcpp::Named("py_values")        = a3_values,
            Rcpp::Named("py_sum")           = py_sum,
            Rcpp::Named("py_d0_equals_crp") = a3_reduces_to_crp,
            Rcpp::Named("tol_exact")        = TOL_EXACT),
        Rcpp::Named("B_crp") = Rcpp::List::create(
            Rcpp::Named("empirical") = crp_emp,
            Rcpp::Named("exact")     = crp_exact,
            Rcpp::Named("tol_4se")   = crp_tol,
            Rcpp::Named("n_draws")   = static_cast<int>(N_CAT)),
        Rcpp::Named("B_py") = Rcpp::List::create(
            Rcpp::Named("empirical") = py_emp,
            Rcpp::Named("exact")     = py_exact,
            Rcpp::Named("tol_4se")   = py_tol,
            Rcpp::Named("n_draws")   = static_cast<int>(N_CAT)),
        Rcpp::Named("C1_k5_n50")   = alpha_result_to_list(c1),
        Rcpp::Named("C2_k20_n100") = alpha_result_to_list(c2));
}

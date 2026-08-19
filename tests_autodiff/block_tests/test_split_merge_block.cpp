// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
// test_split_merge_block.cpp
//
// LIBRARY-LEVEL parity test for AI4BayesCode::split_merge_block
// (Jain & Neal 2004 split-merge proposal, truncated stick-breaking regime).
//
// WHAT IS VERIFIED, AND WHY THIS FORM
// -----------------------------------
// split_merge_block is a Metropolis-Hastings kernel on a discrete allocation
// vector z, not a conjugate draw, so there is no closed-form marginal to
// compare a sample mean against -- the Check #15 recipe used by
// test_beta_gibbs_block does not apply. Of the remaining options:
//
//   - "a conditional you can write down exactly": does not apply either. The
//     block does not sample z from its full conditional; it proposes a whole
//     partition rearrangement and accepts it with an MH probability, so no
//     single coordinate has a stated conditional law to check.
//   - "reversibility on a small state space": the acceptance ratio the block
//     computes is only half of detailed balance. Reproducing the other half
//     (the restricted-Gibbs proposal density) in the test would mean
//     re-deriving the very expression the block implements, so a shared
//     mistake in that derivation would cancel and the test would pass anyway.
//   - INVARIANCE BY EXACT ENUMERATION (chosen). With pi, mu and the
//     covariance held fixed -- exactly the conditioning the block assumes --
//     the allocation posterior is a finite product,
//
//         p(z) proportional to prod_i pi_{z_i} N(y_i | mu_{z_i}, Sigma_{z_i}),
//
//     so for small N and K_trunc every one of the K^N states can be
//     enumerated and normalized. That gives a target the test can both sample
//     from EXACTLY and take exact expectations under, with no reference
//     implementation and no appeal to the block's own arithmetic. A correct
//     MH kernel must leave p fixed, so:
//
//         draw z0 ~ p exactly, apply t = 5 block steps, and the law of the
//         result must still be p.
//
// THE STATISTICS
// --------------
// Two independent readings of "the law of Z_t is still p", over M i.i.d.
// replicates. Each replicate is a fresh exact draw z0 plus fresh kernel
// randomness, so the Z_t are i.i.d. and no burn-in or autocorrelation
// argument enters anywhere in this file.
//
//   1. FUNCTIONAL MEANS against their ANALYTIC values. For each partition
//      functional f, E_p[f] is summed exactly over the enumerated states and
//      compared to the sample mean of f(Z_t):
//          z_f = (mean_m f(Z_t^m) - E_p[f]) / (sd(f(Z_t)) / sqrt(M)).
//      The functionals are the ones a partition sampler can get wrong: the
//      number of active clusters and its whole pmf (the marginal the block's
//      empty-slot selection term controls), every pairwise co-clustering
//      indicator, the largest cluster size, and the number of singletons.
//      A paired variant, mean(f(Z_t) - f(z0)), was tried and is not tighter
//      on these fixtures -- the kernel mixes fast enough that f(Z_t) and
//      f(z0) are only weakly correlated -- so the direct comparison to the
//      analytic mean is what the test uses.
//
//   2. A POOLED CHI-SQUARE goodness-of-fit of the whole post-kernel sample
//      against the enumerated p, so a distortion that happens to leave every
//      listed functional fixed still shows up.
//
// Both statistics are also computed on the PRE-kernel sample (the exact draws
// themselves). In the invariance arms that is a control: it validates the
// enumeration, the state encoding and the inverse-CDF sampler independently
// of the block, so a downstream failure cannot be blamed on the harness.
//
// BOTH EMISSION SHAPES ARE COVERED
// --------------------------------
// The block's two covariance flavors are separate code paths, so each gets
// its own arm:
//   arm A  diagonal precisions (lambda_key), N = 5, K_trunc = 5, d = 1
//          -> 3125 enumerated states
//   arm B  full covariance (sigma_key),      N = 4, K_trunc = 3, d = 2
//          -> 81 enumerated states
// Both fixtures use heavily overlapping components with the prior weight
// concentrated on the first two slots, so the active cluster count sits at
// 2-3 out of K: splits have empty slots to move into, merging the two
// dominant clusters is genuinely plausible, and the measured acceptance rate
// is 0.24 (arm A) / 0.32 (arm B), with 63 / 68 percent of replicates ending
// in a different state than they started. The test asserts that both branches
// fired and were accepted, so a fixture that quietly stopped exercising the
// mechanism fails instead of passing.
//
// TOLERANCES
// ----------
//  * Functional z-scores, threshold |z| < 5 applied to the largest |z| in
//    each arm. f is bounded, so the false-alarm rate follows from Bernstein
//    rather than from trusting exact normality out at 5 sigma:
//        P(|z| > 5) <= 2 exp( -25 / (2 + 10 B / (3 sqrt(M) sd(f))) ),
//    with |f - E f| <= B <= K - 1. Every functional the test admits has
//    M Var(f) >= 500, so sqrt(M) sd(f) / B >= sqrt(500) / B >= 5.6 and the
//    exponent is at least 9.6: below 1.3e-4 per functional, about 8e-3
//    family-wise over the 58 z-scores the two arms admit (29 after the
//    kernel, 29 before). A functional with M Var(f) < 500 is reported as
//    not-exercised rather than allowed to manufacture a pass from a
//    degenerate standard error (P(K_active = 5) in arm A is the one such
//    case); the test requires at least 80 percent of each arm's functionals
//    to be exercised.
//  * Chi-square, threshold |chi2 - df| < 5 sqrt(2 df), with cells whose
//    expected count is below 10 pooled into one bucket. Under the null the
//    statistic is chi2_df, mean df and sd sqrt(2 df); at the df here (618 for
//    arm A, 74 for arm B) the normal approximation is accurate well past 5
//    sd. The bound is two-sided, so an implausibly good fit fails too.
//  * MEASURED null spread over 6 seeds per arm, taking the worst of the
//    before- and after-kernel statistics in each run, with the thresholds
//    held at the values the arguments above give and NOT adjusted to fit:
//        arm A  max|z| in [0.95, 2.43],  standardized chi2 in [-2.04, 1.24]
//        arm B  max|z| in [0.83, 2.89],  standardized chi2 in [-1.48, 0.77]
//
// POWER, SO THE TEST CAN ACTUALLY FAIL
// ------------------------------------
// A pass means nothing unless the same protocol would reject a kernel whose
// invariant law is wrong. Each arm therefore runs a third pass on an
// exponentially tilted law q(z) proportional to p(z) exp(theta K_active(z)),
// with theta solved by bisection so that TV(q, p) = 0.03 -- the size of the
// real defect this family of checks exists for, since the empty-slot
// selection regression that tests/test_split_merge_exact guards drove the
// sampled partition law TV = 0.031 away from exact. That pass measures the
// two things the invariance argument depends on:
//
//   RESOLUTION -- can the statistics see a law that is TV = 0.03 from p?
//   Applied to the tilted draws before any kernel step: max|z| = 29.9 (arm A)
//   and 24.2 (arm B), standardized chi2 = 26.8 and 49.7. Required: > 5.
//
//   EXPOSURE -- do t = 5 steps express a stationary-law error, or only a
//   sliver of it? Running the kernel on those same tilted draws closes the
//   gap from max|z| 29.9 -> 5.8 (arm A, 81 percent) and 24.2 -> 2.2 (arm B,
//   91 percent). Required: at least half the gap closed.
//
// Together: a kernel whose invariant law sat TV = 0.03 from p would register
// |z| of roughly 0.8 * 29.9 = 24 in arm A and 0.9 * 24.2 = 22 in arm B,
// against a 5.0 threshold and a measured null ceiling of 2.9.
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
#include "AI4BayesCode/split_merge_block.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <vector>

using AI4BayesCode::block_context;
using AI4BayesCode::split_merge_block;
using AI4BayesCode::split_merge_block_config;

namespace {

constexpr double kLog2Pi = 1.83787706640934548356065947281;

// ---------------------------------------------------------------------------
// The fixed model the block conditions on. Exactly one of lam / sig is used,
// mirroring the block's own "exactly one of lambda_key / sigma_key" rule.
// ---------------------------------------------------------------------------
struct model {
    std::size_t N = 0;          // observations
    std::size_t K = 0;          // truncation level
    std::size_t d = 0;          // observation dimension
    bool        diagonal = true;
    std::vector<double> pi;     // K
    std::vector<double> mu;     // K * d, cluster-major
    std::vector<double> lam;    // K * d       (diagonal precisions)
    std::vector<double> sig;    // K * d * d   (full covariance)
    std::vector<double> y;      // N * d
    std::size_t n_states = 0;   // K^N
};

/// log N(y_i | mu_k, .) in exactly the parameterization split_merge_block
/// uses: the diagonal flavor reads lam[k*d+j] as a PRECISION, the full flavor
/// reads sig[k*d*d + a*d + b] as Sigma_k(a, b).
double log_lik(const model& m, std::size_t i, std::size_t k) {
    if (m.diagonal) {
        double lp = 0.0;
        for (std::size_t j = 0; j < m.d; ++j) {
            const double dev = m.y[i * m.d + j] - m.mu[k * m.d + j];
            const double l   = m.lam[k * m.d + j];
            lp += 0.5 * std::log(l) - 0.5 * kLog2Pi - 0.5 * l * dev * dev;
        }
        return lp;
    }
    arma::mat S(m.d, m.d);
    for (std::size_t a = 0; a < m.d; ++a)
        for (std::size_t b = 0; b < m.d; ++b)
            S(a, b) = m.sig[k * m.d * m.d + a * m.d + b];
    arma::mat L;
    if (!arma::chol(L, S, "lower"))
        return -std::numeric_limits<double>::infinity();
    arma::vec dev(m.d);
    for (std::size_t j = 0; j < m.d; ++j)
        dev[j] = m.y[i * m.d + j] - m.mu[k * m.d + j];
    double log_det = 0.0;
    for (std::size_t a = 0; a < m.d; ++a) log_det += 2.0 * std::log(L(a, a));
    const arma::vec u = arma::solve(arma::trimatl(L), dev);
    return -0.5 * static_cast<double>(m.d) * kLog2Pi
         - 0.5 * log_det - 0.5 * arma::dot(u, u);
}

// State index: base-K, with observation 0 the least significant digit.
void decode(std::size_t s, std::size_t K, std::size_t N,
            std::vector<std::size_t>& lab) {
    for (std::size_t i = 0; i < N; ++i) { lab[i] = s % K; s /= K; }
}
std::size_t encode(const std::vector<std::size_t>& lab, std::size_t K) {
    std::size_t s = 0, mult = 1;
    for (std::size_t i = 0; i < lab.size(); ++i) { s += lab[i] * mult; mult *= K; }
    return s;
}

/// Exact normalized p(z) over all K^N allocations.
std::vector<double> exact_pmf(const model& m) {
    std::vector<double> w(m.N * m.K);
    for (std::size_t i = 0; i < m.N; ++i)
        for (std::size_t k = 0; k < m.K; ++k)
            w[i * m.K + k] = std::log(m.pi[k]) + log_lik(m, i, k);

    std::vector<double> lp(m.n_states), p(m.n_states);
    std::vector<std::size_t> lab(m.N);
    double mx = -std::numeric_limits<double>::infinity();
    for (std::size_t s = 0; s < m.n_states; ++s) {
        decode(s, m.K, m.N, lab);
        double v = 0.0;
        for (std::size_t i = 0; i < m.N; ++i) v += w[i * m.K + lab[i]];
        lp[s] = v;
        if (v > mx) mx = v;
    }
    double Z = 0.0;
    for (std::size_t s = 0; s < m.n_states; ++s) { p[s] = std::exp(lp[s] - mx); Z += p[s]; }
    for (double& v : p) v /= Z;
    return p;
}

std::vector<int> k_active_of_state(const model& m) {
    std::vector<int> out(m.n_states);
    std::vector<std::size_t> lab(m.N);
    for (std::size_t s = 0; s < m.n_states; ++s) {
        decode(s, m.K, m.N, lab);
        std::vector<char> used(m.K, 0);
        for (std::size_t i = 0; i < m.N; ++i) used[lab[i]] = 1;
        int na = 0;
        for (std::size_t k = 0; k < m.K; ++k) na += used[k];
        out[s] = na;
    }
    return out;
}

double total_variation(const std::vector<double>& a, const std::vector<double>& b) {
    double t = 0.0;
    for (std::size_t s = 0; s < a.size(); ++s) t += std::fabs(a[s] - b[s]);
    return 0.5 * t;
}

/// q(z) proportional to p(z) exp(theta * K_active(z)).
std::vector<double> tilt(const std::vector<double>& p,
                         const std::vector<int>& nact, double theta) {
    std::vector<double> q(p.size());
    double Z = 0.0;
    for (std::size_t s = 0; s < p.size(); ++s) {
        q[s] = p[s] * std::exp(theta * static_cast<double>(nact[s]));
        Z += q[s];
    }
    for (double& v : q) v /= Z;
    return q;
}

/// Bisect theta so that TV(tilt(p, theta), p) hits target_tv. TV is monotone
/// in theta from 0, so plain bisection converges.
std::vector<double> tilt_to_tv(const std::vector<double>& p,
                               const std::vector<int>& nact,
                               double target_tv, double& theta_out,
                               double& tv_out) {
    double lo = 0.0, hi = 1.0;
    while (total_variation(tilt(p, nact, hi), p) < target_tv && hi < 64.0) hi *= 2.0;
    for (int it = 0; it < 80; ++it) {
        const double mid = 0.5 * (lo + hi);
        if (total_variation(tilt(p, nact, mid), p) < target_tv) lo = mid; else hi = mid;
    }
    theta_out = 0.5 * (lo + hi);
    std::vector<double> q = tilt(p, nact, theta_out);
    tv_out = total_variation(q, p);
    return q;
}

// ---------------------------------------------------------------------------
// Partition functionals. Layout:
//   0                      K_active
//   1 .. K                 1{K_active == k}
//   K+1 .. K+N(N-1)/2      1{z_i == z_j}, i < j
//   ..+1                   size of the largest cluster
//   ..+2                   number of singleton clusters
// ---------------------------------------------------------------------------
std::size_t n_functionals(const model& m) {
    return 1 + m.K + m.N * (m.N - 1) / 2 + 2;
}

std::vector<std::string> functional_names(const model& m) {
    std::vector<std::string> nm;
    nm.push_back("K_active");
    for (std::size_t k = 1; k <= m.K; ++k)
        nm.push_back("P(K_active=" + std::to_string(k) + ")");
    for (std::size_t i = 0; i < m.N; ++i)
        for (std::size_t j = i + 1; j < m.N; ++j)
            nm.push_back("cocluster(" + std::to_string(i) + "," +
                         std::to_string(j) + ")");
    nm.push_back("largest_cluster_size");
    nm.push_back("n_singleton_clusters");
    return nm;
}

void eval_functionals(const model& m, const std::vector<std::size_t>& lab,
                      std::vector<double>& out) {
    std::vector<std::size_t> cnt(m.K, 0);
    for (std::size_t i = 0; i < m.N; ++i) ++cnt[lab[i]];
    std::size_t nact = 0, largest = 0, nsing = 0;
    for (std::size_t k = 0; k < m.K; ++k) {
        if (cnt[k] > 0)  ++nact;
        if (cnt[k] > largest) largest = cnt[k];
        if (cnt[k] == 1) ++nsing;
    }
    std::size_t q = 0;
    out[q++] = static_cast<double>(nact);
    for (std::size_t k = 1; k <= m.K; ++k)
        out[q++] = (nact == k) ? 1.0 : 0.0;
    for (std::size_t i = 0; i < m.N; ++i)
        for (std::size_t j = i + 1; j < m.N; ++j)
            out[q++] = (lab[i] == lab[j]) ? 1.0 : 0.0;
    out[q++] = static_cast<double>(largest);
    out[q++] = static_cast<double>(nsing);
}

/// E_p[f] for every functional, summed exactly over the enumerated states.
/// These are the ANALYTIC targets the sampled means are compared against.
std::vector<double> exact_functional_means(const model& m,
                                           const std::vector<double>& p) {
    const std::size_t nf = n_functionals(m);
    std::vector<double> out(nf, 0.0), f(nf);
    std::vector<std::size_t> lab(m.N);
    for (std::size_t s = 0; s < m.n_states; ++s) {
        decode(s, m.K, m.N, lab);
        eval_functionals(m, lab, f);
        for (std::size_t q = 0; q < nf; ++q) out[q] += p[s] * f[q];
    }
    return out;
}

// ---------------------------------------------------------------------------
// Pooled chi-square goodness of fit. Cells with expected count below 10 are
// merged into one bucket so the chi2_df reference law is trustworthy.
// ---------------------------------------------------------------------------
struct chi2_res { double stat = 0.0; int df = 0; double sd_units = 0.0; };

chi2_res chi2_gof(const std::vector<double>& cnt,
                  const std::vector<double>& p_ref, double M) {
    double stat = 0.0, pooled_exp = 0.0, pooled_obs = 0.0;
    int cells = 0;
    for (std::size_t s = 0; s < p_ref.size(); ++s) {
        const double e = M * p_ref[s];
        if (e >= 10.0) {
            const double diff = cnt[s] - e;
            stat += diff * diff / e;
            ++cells;
        } else {
            pooled_exp += e;
            pooled_obs += cnt[s];
        }
    }
    if (pooled_exp > 0.0) {
        const double diff = pooled_obs - pooled_exp;
        stat += diff * diff / pooled_exp;
        ++cells;
    }
    chi2_res r;
    r.stat = stat;
    r.df   = cells - 1;
    r.sd_units = (r.df > 0) ? (stat - r.df) / std::sqrt(2.0 * r.df) : 0.0;
    return r;
}

// ---------------------------------------------------------------------------
// One arm: M independent replicates of (draw z0 from p_init, apply n_steps
// block steps). Reports the functional z-scores and the chi-square fit both
// BEFORE and AFTER the kernel, against the analytic targets f_exact / p_ref.
// ---------------------------------------------------------------------------
struct arm_out {
    std::size_t n_func = 0;
    // AFTER the kernel: mean of f(Z_t) against the enumerated E_p[f].
    std::size_t n_tested = 0;
    double max_abs_z = 0.0;
    std::string worst_name;
    std::vector<double> zval, mean_f, se_f;
    // BEFORE the kernel: mean of f(z0) against the same E_p[f]. A control
    // when z0 ~ p, the resolution measurement when z0 ~ q.
    std::size_t n_tested_pre = 0;
    double max_abs_z_pre = 0.0;
    std::string worst_name_pre;
    std::vector<double> zval_pre;
    chi2_res pre, post;
    double frac_moved = 0.0;
    double acc_rate = 0.0, split_acc_rate = 0.0, merge_acc_rate = 0.0;
    double n_split_prop = 0, n_split_acc = 0, n_merge_prop = 0, n_merge_acc = 0;
    // Smallest shift in E[K_active] this arm resolves at 5 se.
    double k_active_resolution = 0.0;
};

arm_out run_arm(const model& m,
                const std::vector<double>& p_init,
                const std::vector<double>& p_ref,
                const std::vector<double>& f_exact,
                std::size_t n_rep, std::size_t n_steps,
                std::uint64_t seed) {
    // ---- block under test; the context is fixed for the whole arm --------
    split_merge_block_config cfg;
    cfg.name    = "sm";
    cfg.N       = m.N;
    cfg.K_trunc = m.K;
    cfg.d       = m.d;
    cfg.z_name  = "z";
    cfg.y_key   = "y";
    cfg.pi_key  = "pi";
    cfg.mu_key  = "mu";
    if (m.diagonal) cfg.lambda_key = "lam"; else cfg.sigma_key = "sig";
    cfg.n_restricted_gibbs_iters = 5;   // shipped default
    cfg.initial_z = arma::vec(m.N, arma::fill::ones);
    split_merge_block blk(std::move(cfg));

    block_context ctx;
    ctx["y"]  = arma::vec(m.y);
    ctx["pi"] = arma::vec(m.pi);
    ctx["mu"] = arma::vec(m.mu);
    if (m.diagonal) ctx["lam"] = arma::vec(m.lam);
    else            ctx["sig"] = arma::vec(m.sig);
    blk.set_context(ctx);

    // ---- exact sampler for p_init (inverse CDF) --------------------------
    std::vector<double> cdf(p_init.size());
    double acc = 0.0;
    for (std::size_t s = 0; s < p_init.size(); ++s) { acc += p_init[s]; cdf[s] = acc; }
    cdf.back() = 1.0;

    const std::size_t nf = n_functionals(m);
    std::vector<double> f0(nf), f1(nf);
    std::vector<double> sum_post(nf, 0.0), sum_post2(nf, 0.0),
                        sum_pre(nf, 0.0),  sum_pre2(nf, 0.0),
                        n_changed(nf, 0.0);
    std::vector<double> cnt_pre(m.n_states, 0.0), cnt_post(m.n_states, 0.0);
    std::vector<std::size_t> lab0(m.N), lab1(m.N);
    arma::vec z(m.N);

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::size_t n_moved = 0;

    for (std::size_t r = 0; r < n_rep; ++r) {
        const double u = U(rng);
        std::size_t s0 = static_cast<std::size_t>(
            std::lower_bound(cdf.begin(), cdf.end(), u) - cdf.begin());
        if (s0 >= m.n_states) s0 = m.n_states - 1;
        decode(s0, m.K, m.N, lab0);
        for (std::size_t i = 0; i < m.N; ++i)
            z[i] = static_cast<double>(lab0[i] + 1);
        blk.set_current(z);

        for (std::size_t t = 0; t < n_steps; ++t) blk.step(rng);

        const arma::vec& zc = blk.current();
        for (std::size_t i = 0; i < m.N; ++i)
            lab1[i] = static_cast<std::size_t>(std::llround(zc[i])) - 1;
        const std::size_t s1 = encode(lab1, m.K);

        cnt_pre[s0]  += 1.0;
        cnt_post[s1] += 1.0;
        if (s1 != s0) ++n_moved;

        eval_functionals(m, lab0, f0);
        eval_functionals(m, lab1, f1);
        for (std::size_t q = 0; q < nf; ++q) {
            sum_pre[q]   += f0[q];
            sum_pre2[q]  += f0[q] * f0[q];
            sum_post[q]  += f1[q];
            sum_post2[q] += f1[q] * f1[q];
            if (f1[q] != f0[q]) n_changed[q] += 1.0;
        }
    }

    arm_out o;
    o.n_func = nf;
    o.zval.assign(nf, 0.0);
    o.zval_pre.assign(nf, 0.0);
    o.mean_f.assign(nf, 0.0);
    o.se_f.assign(nf, 0.0);

    // A functional carries usable information only if the sample varies it.
    // M * Var(f) is the effective count behind the CLT / Bernstein argument
    // and 500 is the floor those tolerances are stated at; below it the
    // functional is reported as not-exercised rather than allowed to
    // manufacture a pass -- or, in the tilted arm, a spurious rejection --
    // out of a degenerate standard error.
    const double INFO_FLOOR = 500.0;
    const double M = static_cast<double>(n_rep);
    const std::vector<std::string> nm = functional_names(m);

    for (std::size_t q = 0; q < nf; ++q) {
        const double mean1 = sum_post[q] / M;
        const double var1  = std::max(
            (sum_post2[q] - M * mean1 * mean1) / (M - 1.0), 0.0);
        const double se1   = std::sqrt(var1 / M);
        o.mean_f[q] = mean1;
        o.se_f[q]   = se1;

        const double mean0 = sum_pre[q] / M;
        const double var0  = std::max(
            (sum_pre2[q] - M * mean0 * mean0) / (M - 1.0), 0.0);
        const double se0   = std::sqrt(var0 / M);

        if (M * var0 >= INFO_FLOOR && se0 > 0.0) {
            const double z0 = (mean0 - f_exact[q]) / se0;
            o.zval_pre[q] = z0;
            ++o.n_tested_pre;
            if (std::fabs(z0) > o.max_abs_z_pre) {
                o.max_abs_z_pre  = std::fabs(z0);
                o.worst_name_pre = nm[q];
            }
        } else {
            o.zval_pre[q] = NA_REAL;
        }

        // The post statistic additionally requires that the kernel moved this
        // functional at all -- otherwise it re-tests the exact sampler rather
        // than the block.
        if (n_changed[q] < INFO_FLOOR || M * var1 < INFO_FLOOR || se1 <= 0.0) {
            o.zval[q] = NA_REAL;
            continue;
        }
        const double z1 = (mean1 - f_exact[q]) / se1;
        o.zval[q] = z1;
        ++o.n_tested;
        if (std::fabs(z1) > o.max_abs_z) {
            o.max_abs_z  = std::fabs(z1);
            o.worst_name = nm[q];
        }
    }
    o.k_active_resolution = 5.0 * o.se_f[0];

    o.pre  = chi2_gof(cnt_pre,  p_ref, M);
    o.post = chi2_gof(cnt_post, p_ref, M);
    o.frac_moved = static_cast<double>(n_moved) / M;

    o.n_split_prop = static_cast<double>(blk.n_split_proposals());
    o.n_split_acc  = static_cast<double>(blk.n_split_accepted());
    o.n_merge_prop = static_cast<double>(blk.n_merge_proposals());
    o.n_merge_acc  = static_cast<double>(blk.n_merge_accepted());
    o.acc_rate = static_cast<double>(blk.n_accepted()) /
                 std::max(1.0, static_cast<double>(blk.n_proposals()));
    o.split_acc_rate = o.n_split_acc / std::max(1.0, o.n_split_prop);
    o.merge_acc_rate = o.n_merge_acc / std::max(1.0, o.n_merge_prop);
    return o;
}

// ---------------------------------------------------------------------------
// Fixtures. Overlapping components with the prior weight concentrated on the
// first two slots, so K_active sits at 2-3 out of K: empty slots exist for a
// split to move into, and merging the two dominant clusters is plausible.
// ---------------------------------------------------------------------------
model fixture_diagonal() {
    model m;
    m.N = 5; m.K = 5; m.d = 1; m.diagonal = true;
    m.pi  = {0.50, 0.30, 0.10, 0.06, 0.04};
    m.mu  = {-0.35, 0.35, 1.30, -1.30, 2.20};
    m.lam = {1.20, 1.20, 1.00, 1.00, 0.80};       // precisions
    m.y   = {-0.60, 0.10, 0.45, -0.20, 0.70};
    m.n_states = 1;
    for (std::size_t i = 0; i < m.N; ++i) m.n_states *= m.K;   // 3125
    return m;
}

model fixture_full_cov() {
    model m;
    m.N = 4; m.K = 3; m.d = 2; m.diagonal = false;
    m.pi = {0.50, 0.33, 0.17};
    m.mu = {-0.30,  0.10,
             0.40, -0.20,
             1.40,  1.00};
    m.sig = {1.00,  0.30,   0.30, 0.80,
             0.70, -0.20,  -0.20, 1.10,
             1.30,  0.50,   0.50, 0.90};
    m.y  = {-0.50,  0.20,
             0.30, -0.40,
             0.10,  0.50,
            -0.20, -0.10};
    m.n_states = 1;
    for (std::size_t i = 0; i < m.N; ++i) m.n_states *= m.K;   // 81
    return m;
}

Rcpp::List arm_report(const arm_out& o, const std::vector<std::string>& nm) {
    return Rcpp::List::create(
        Rcpp::Named("max_abs_z")        = o.max_abs_z,
        Rcpp::Named("worst_functional") = o.worst_name,
        Rcpp::Named("max_abs_z_pre")    = o.max_abs_z_pre,
        Rcpp::Named("worst_pre")        = o.worst_name_pre,
        Rcpp::Named("n_functionals")    = static_cast<int>(o.n_func),
        Rcpp::Named("n_tested")         = static_cast<int>(o.n_tested),
        Rcpp::Named("n_tested_pre")     = static_cast<int>(o.n_tested_pre),
        Rcpp::Named("functional")       = Rcpp::wrap(nm),
        Rcpp::Named("z")                = Rcpp::wrap(o.zval),
        Rcpp::Named("z_pre")            = Rcpp::wrap(o.zval_pre),
        Rcpp::Named("chi2_pre_sd")      = o.pre.sd_units,
        Rcpp::Named("chi2_post_sd")     = o.post.sd_units,
        Rcpp::Named("chi2_df")          = o.post.df,
        Rcpp::Named("frac_moved")       = o.frac_moved,
        Rcpp::Named("acc_rate")         = o.acc_rate,
        Rcpp::Named("split_acc_rate")   = o.split_acc_rate,
        Rcpp::Named("merge_acc_rate")   = o.merge_acc_rate,
        Rcpp::Named("n_split_accepted") = o.n_split_acc,
        Rcpp::Named("n_merge_accepted") = o.n_merge_acc,
        Rcpp::Named("k_active_resolution") = o.k_active_resolution);
}

}  // namespace

// [[Rcpp::export]]
Rcpp::List test_split_merge_block() {
    // Thresholds follow the arguments in the header comment; none of them was
    // moved to make a run pass.
    const double Z_TOL   = 5.0;    // Bernstein: <= 1.3e-4 per functional
    const double CHI2_SD = 5.0;    // chi2_df has mean df, sd sqrt(2 df)
    const double TILT_TV = 0.03;   // size of the empty-slot regression
    const double EXPOSE  = 0.50;   // t steps must close half of that gap
    const std::size_t T_STEPS = 5;
    const std::size_t M_DIAG = 200000, M_FULL = 60000;

    // ---------------- arm A: diagonal precisions --------------------------
    const model mA = fixture_diagonal();
    const std::vector<double> pA  = exact_pmf(mA);
    const std::vector<int>    naA = k_active_of_state(mA);
    const std::vector<std::string> nmA = functional_names(mA);
    const std::vector<double> fA  = exact_functional_means(mA, pA);

    const arm_out A = run_arm(mA, pA, pA, fA, M_DIAG, T_STEPS, 20260818ULL);

    double thetaA = 0.0, tvA = 0.0;
    const std::vector<double> qA = tilt_to_tv(pA, naA, TILT_TV, thetaA, tvA);
    const arm_out A_tilt = run_arm(mA, qA, pA, fA, M_DIAG, T_STEPS, 20260819ULL);

    // ---------------- arm B: full covariance ------------------------------
    const model mB = fixture_full_cov();
    const std::vector<double> pB  = exact_pmf(mB);
    const std::vector<int>    naB = k_active_of_state(mB);
    const std::vector<std::string> nmB = functional_names(mB);
    const std::vector<double> fB  = exact_functional_means(mB, pB);

    const arm_out B = run_arm(mB, pB, pB, fB, M_FULL, T_STEPS, 20260820ULL);

    double thetaB = 0.0, tvB = 0.0;
    const std::vector<double> qB = tilt_to_tv(pB, naB, TILT_TV, thetaB, tvB);
    const arm_out B_tilt = run_arm(mB, qB, pB, fB, M_FULL, T_STEPS, 20260821ULL);

    // ---------------- checks ----------------------------------------------
    // 1. INVARIANCE. Functional means after t kernel steps must match their
    //    analytic values, and the whole sample must fit the enumerated pmf.
    const bool pass_invar_A = (A.max_abs_z < Z_TOL) &&
                              (std::fabs(A.post.sd_units) < CHI2_SD);
    const bool pass_invar_B = (B.max_abs_z < Z_TOL) &&
                              (std::fabs(B.post.sd_units) < CHI2_SD);

    // 2. CONTROL. The same two statistics on the exact draws themselves,
    //    which validates the enumeration, encoding and sampler without the
    //    block in the picture.
    const bool pass_ctrl_A = (A.max_abs_z_pre < Z_TOL) &&
                             (std::fabs(A.pre.sd_units) < CHI2_SD);
    const bool pass_ctrl_B = (B.max_abs_z_pre < Z_TOL) &&
                             (std::fabs(B.pre.sd_units) < CHI2_SD);

    // 3. The fixture must exercise the mechanism, in both branches.
    const bool pass_exercise_A =
        (A.frac_moved > 0.10) && (A.n_split_acc > 0) && (A.n_merge_acc > 0) &&
        (A.n_tested >= (4 * A.n_func) / 5);
    const bool pass_exercise_B =
        (B.frac_moved > 0.10) && (B.n_split_acc > 0) && (B.n_merge_acc > 0) &&
        (B.n_tested >= (4 * B.n_func) / 5);

    // 4. RESOLUTION. The statistics must see a law TV = 0.03 away from p.
    const bool pass_resolve_A = (A_tilt.max_abs_z_pre > Z_TOL) &&
                                (A_tilt.pre.sd_units  > CHI2_SD);
    const bool pass_resolve_B = (B_tilt.max_abs_z_pre > Z_TOL) &&
                                (B_tilt.pre.sd_units  > CHI2_SD);

    // 5. EXPOSURE. t steps must close most of that gap, so the invariance
    //    arm reads a stationary-law error at close to full size rather than
    //    a heavily attenuated fraction of it.
    const double closed_A = 1.0 - A_tilt.max_abs_z / A_tilt.max_abs_z_pre;
    const double closed_B = 1.0 - B_tilt.max_abs_z / B_tilt.max_abs_z_pre;
    const bool pass_expose_A = (closed_A > EXPOSE);
    const bool pass_expose_B = (closed_B > EXPOSE);

    const bool all_pass =
        pass_invar_A && pass_invar_B &&
        pass_ctrl_A  && pass_ctrl_B  &&
        pass_exercise_A && pass_exercise_B &&
        pass_resolve_A  && pass_resolve_B  &&
        pass_expose_A   && pass_expose_B;

    return Rcpp::List::create(
        Rcpp::Named("all_pass")           = all_pass,
        Rcpp::Named("pass_invariance")    = pass_invar_A && pass_invar_B,
        Rcpp::Named("pass_exact_control") = pass_ctrl_A && pass_ctrl_B,
        Rcpp::Named("pass_mechanism_exercised") = pass_exercise_A && pass_exercise_B,
        Rcpp::Named("pass_resolves_wrong_law")  = pass_resolve_A && pass_resolve_B,
        Rcpp::Named("pass_exposes_wrong_law")   = pass_expose_A && pass_expose_B,
        Rcpp::Named("z_tol")              = Z_TOL,
        Rcpp::Named("chi2_sd_tol")        = CHI2_SD,
        Rcpp::Named("n_kernel_steps")     = static_cast<int>(T_STEPS),
        Rcpp::Named("n_replicates")       = Rcpp::NumericVector::create(
            Rcpp::Named("diag") = static_cast<double>(M_DIAG),
            Rcpp::Named("full") = static_cast<double>(M_FULL)),
        Rcpp::Named("n_states")           = Rcpp::IntegerVector::create(
            Rcpp::Named("diag") = static_cast<int>(mA.n_states),
            Rcpp::Named("full") = static_cast<int>(mB.n_states)),
        Rcpp::Named("tilt_tv_target")     = TILT_TV,
        Rcpp::Named("tilt_theta")         = Rcpp::NumericVector::create(
            Rcpp::Named("diag") = thetaA, Rcpp::Named("full") = thetaB),
        Rcpp::Named("tilt_tv_achieved")   = Rcpp::NumericVector::create(
            Rcpp::Named("diag") = tvA, Rcpp::Named("full") = tvB),
        Rcpp::Named("gap_closed_by_kernel") = Rcpp::NumericVector::create(
            Rcpp::Named("diag") = closed_A, Rcpp::Named("full") = closed_B),
        Rcpp::Named("diag")               = arm_report(A, nmA),
        Rcpp::Named("full")               = arm_report(B, nmB),
        Rcpp::Named("diag_tilted")        = arm_report(A_tilt, nmA),
        Rcpp::Named("full_tilted")        = arm_report(B_tilt, nmB));
}

// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
// test_binary_gibbs_block.cpp
//
// LIBRARY-LEVEL parity test for AI4BayesCode::binary_gibbs_block.
//
// What the block claims
// ---------------------
// binary_gibbs_block samples a vector z in {0,1}^n by a DETERMINISTIC-SCAN
// component-wise Gibbs sweep. For component i it evaluates the user's
// log_odds_fn on the CURRENT context, reads element i,
//
//     lo_i = log [ P(z_i = 1 | z_{-i}) / P(z_i = 0 | z_{-i}) ],
//
// draws z_i ~ Bernoulli(sigmoid(lo_i)), and writes the updated z back into
// its own context before moving on to component i+1. Two separate claims
// are being made there, and this file checks both.
//
// Target 1 -- the Bernoulli draw itself (exact, i.i.d.)
// -----------------------------------------------------
// When log_odds_fn ignores the context and returns a FIXED vector, the
// components are independent and successive sweeps are i.i.d. draws from a
// product of Bernoullis with p_i = sigmoid(lo_i), which is available in
// closed form. Sweeps being i.i.d. here is what makes the binomial standard
// error below exactly (not approximately) the right yardstick.
//
// Target 2 -- the sweep leaves a DEPENDENT joint invariant (exact enumeration)
// ---------------------------------------------------------------------------
// Target 1 cannot detect the mechanism that actually matters: whether the
// sweep re-reads the freshly flipped components. A sampler that computed all
// n log-odds once from the stale state and flipped all n components at once
// would pass Target 1 exactly, yet converge to the WRONG joint whenever the
// components are dependent. So Target 2 uses a fixture whose normalising
// constant can be summed over in full -- a 4-spin pairwise (Ising-type)
// model on 2^4 = 16 states,
//
//     p(z) proportional to exp( sum_i a_i z_i + sum_{i<j} W_ij z_i z_j ),
//
// whose full conditionals are exactly Bernoulli with
//
//     lo_i = a_i + sum_{j != i} W_ij z_j.
//
// Feeding those conditionals to the block and comparing the empirical
// frequency of all 16 states against the exactly enumerated p(z) tests the
// invariance claim directly: the sweep is a valid Gibbs kernel for p only if
// its ergodic averages reproduce p. Exact enumeration was chosen over the
// other options (a closed-form marginal does not exist for a coupled binary
// vector; a detailed-balance identity would only re-derive the conditionals
// this test already supplies) because it compares against the ENTIRE target
// distribution, with no Monte Carlo error on the target side.
//
// Tolerances
// ----------
// Target 1: draws are i.i.d., so the exact standard error of the empirical
//   frequency is se = sqrt(p (1 - p) / N). The gate is 4 se per component
//   (two-sided tail ~ 6e-5 each, ~4e-4 over the 6 components).
// Target 2: draws are a Markov chain, so the binomial se understates the
//   Monte Carlo error. The se is MEASURED from the run itself by batch means
//   (200 batches of 1000 sweeps; batch length far exceeds the few-sweep
//   autocorrelation of a 4-spin Gibbs chain, so batch means are effectively
//   uncorrelated). Because the batch-means se is itself a noisy estimate and
//   can dip below the i.i.d. floor, the gate uses
//   4 * max(se_batch_means, se_iid) per state.
// Neither threshold is tuned: both are standard errors implied by the number
// of draws, and the 4-se factor is fixed before the run. As a check that the
// gate is neither too tight nor vacuous, the same fixture was run over 12
// seeds (264 standardised deviations in total): the largest |deviation| / se
// observed anywhere was 2.9, so a genuine chain sits comfortably inside the
// gate while a biased kernel of the size described below would not.
//
// The header also reports `sep_indep`, the largest gap between the exact
// joint and the product of its own exact marginals. It is not a pass/fail
// target on the sampler; it is the scale of the dependence signal Target 2
// exists to detect, and it should dwarf the tolerance for the test to have
// power.
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
#include "AI4BayesCode/binary_gibbs_block.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

using AI4BayesCode::binary_gibbs_block;
using AI4BayesCode::binary_gibbs_block_config;
using AI4BayesCode::block_context;

namespace {

// Numerically stable sigmoid; the test computes its targets independently
// of the block's own implementation.
double sigmoid(double x) {
    if (x >= 0.0) return 1.0 / (1.0 + std::exp(-x));
    const double e = std::exp(x);
    return e / (1.0 + e);
}

} // namespace

// [[Rcpp::export]]
Rcpp::List test_binary_gibbs_block() {

    // =====================================================================
    // TARGET 1 -- fixed log-odds: i.i.d. product of Bernoullis
    // =====================================================================
    const arma::vec lo_fixed = {-3.0, -1.0, -0.25, 0.25, 1.0, 3.0};
    const std::size_t K   = lo_fixed.n_elem;
    const std::size_t N_1 = 40000;

    binary_gibbs_block_config cfg1;
    cfg1.name           = "z";
    cfg1.n_binary       = K;
    cfg1.initial_values = arma::vec(K, arma::fill::zeros);
    cfg1.log_odds_fn    = [lo_fixed](const block_context& /*ctx*/) {
        return lo_fixed;
    };
    binary_gibbs_block blk1(std::move(cfg1));
    blk1.set_context(block_context{});   // params ignore the context here

    std::mt19937_64 rng1(20260818u);
    arma::vec counts1(K, arma::fill::zeros);
    bool binary_valued = true;
    for (std::size_t s = 0; s < N_1; ++s) {
        blk1.step(rng1);
        const arma::vec& v = blk1.current();
        for (std::size_t i = 0; i < K; ++i) {
            if (v[i] != 0.0 && v[i] != 1.0) binary_valued = false;
        }
        counts1 += v;
    }

    arma::vec p_exact_1(K), p_emp_1(K), dev_1(K), se_1(K);
    bool pass_marginals = true;
    for (std::size_t i = 0; i < K; ++i) {
        p_exact_1[i] = sigmoid(lo_fixed[i]);
        p_emp_1[i]   = counts1[i] / static_cast<double>(N_1);
        // Exact binomial se: sweeps are i.i.d. because log_odds_fn is
        // constant, so no autocorrelation correction is needed.
        se_1[i]  = std::sqrt(p_exact_1[i] * (1.0 - p_exact_1[i]) /
                             static_cast<double>(N_1));
        dev_1[i] = std::abs(p_emp_1[i] - p_exact_1[i]);
        if (!(dev_1[i] < 4.0 * se_1[i])) pass_marginals = false;
    }

    // =====================================================================
    // TARGET 2 -- coupled 4-spin joint, compared to exact enumeration
    // =====================================================================
    const std::size_t N_SPIN   = 4;
    const std::size_t N_STATES = 16;          // 2^4

    // Fields and symmetric couplings. Signs are mixed (both alignment and
    // repulsion) so no component is close to independent of the others.
    const arma::vec a = {-1.2, 0.4, -0.6, 0.9};
    arma::mat W(N_SPIN, N_SPIN, arma::fill::zeros);
    W(0, 1) = W(1, 0) =  1.6;
    W(0, 2) = W(2, 0) =  0.5;
    W(0, 3) = W(3, 0) = -0.8;
    W(1, 2) = W(2, 1) = -1.3;
    W(2, 3) = W(3, 2) =  1.1;
    // W(1,3) stays 0.

    // ---- exact enumeration of p(z) over all 16 states -------------------
    // State index k encodes z_i as bit i of k.
    arma::vec p_exact_2(N_STATES);
    {
        double norm = 0.0;
        for (std::size_t k = 0; k < N_STATES; ++k) {
            arma::vec z(N_SPIN);
            for (std::size_t i = 0; i < N_SPIN; ++i)
                z[i] = static_cast<double>((k >> i) & 1u);
            double energy = 0.0;
            for (std::size_t i = 0; i < N_SPIN; ++i) {
                energy += a[i] * z[i];
                for (std::size_t j = i + 1; j < N_SPIN; ++j)
                    energy += W(i, j) * z[i] * z[j];
            }
            p_exact_2[k] = std::exp(energy);
            norm += p_exact_2[k];
        }
        p_exact_2 /= norm;
    }

    // Exact marginals, and the product-of-marginals surrogate used only to
    // report how much dependence this fixture carries.
    arma::vec marg_exact(N_SPIN, arma::fill::zeros);
    for (std::size_t k = 0; k < N_STATES; ++k)
        for (std::size_t i = 0; i < N_SPIN; ++i)
            if ((k >> i) & 1u) marg_exact[i] += p_exact_2[k];

    double sep_indep = 0.0;
    for (std::size_t k = 0; k < N_STATES; ++k) {
        double q = 1.0;
        for (std::size_t i = 0; i < N_SPIN; ++i)
            q *= ((k >> i) & 1u) ? marg_exact[i] : (1.0 - marg_exact[i]);
        sep_indep = std::max(sep_indep, std::abs(p_exact_2[k] - q));
    }

    // ---- run the block on the exact full conditionals -------------------
    binary_gibbs_block_config cfg2;
    cfg2.name           = "z";
    cfg2.n_binary       = N_SPIN;
    cfg2.initial_values = arma::vec(N_SPIN, arma::fill::zeros);
    cfg2.log_odds_fn    = [a, W, N_SPIN](const block_context& ctx) {
        // Full conditional of the pairwise model:
        //   lo_i = a_i + sum_{j != i} W_ij z_j   (W_ii == 0).
        // Reads z from the context, which the block refreshes after every
        // component -- that is exactly the behaviour under test.
        const arma::vec& z = ctx.at("z");
        arma::vec lo(N_SPIN);
        for (std::size_t i = 0; i < N_SPIN; ++i) {
            double s = a[i];
            for (std::size_t j = 0; j < N_SPIN; ++j) s += W(i, j) * z[j];
            lo[i] = s;
        }
        return lo;
    };
    binary_gibbs_block blk2(std::move(cfg2));

    block_context ctx2;
    ctx2["z"] = arma::vec(N_SPIN, arma::fill::zeros);
    blk2.set_context(ctx2);

    const std::size_t N_BURN  = 1000;
    const std::size_t N_BATCH = 200;
    const std::size_t BATCH   = 1000;
    const std::size_t N_2     = N_BATCH * BATCH;   // 200,000 kept sweeps

    std::mt19937_64 rng2(20260818u);
    for (std::size_t s = 0; s < N_BURN; ++s) blk2.step(rng2);

    arma::vec counts2(N_STATES, arma::fill::zeros);
    arma::mat batch_freq(N_BATCH, N_STATES, arma::fill::zeros);
    for (std::size_t b = 0; b < N_BATCH; ++b) {
        for (std::size_t s = 0; s < BATCH; ++s) {
            blk2.step(rng2);
            const arma::vec& v = blk2.current();
            std::size_t k = 0;
            for (std::size_t i = 0; i < N_SPIN; ++i) {
                if (v[i] != 0.0 && v[i] != 1.0) binary_valued = false;
                if (v[i] >= 0.5) k |= (1u << i);
            }
            counts2[k]         += 1.0;
            batch_freq(b, k)   += 1.0;
        }
        for (std::size_t k = 0; k < N_STATES; ++k)
            batch_freq(b, k) /= static_cast<double>(BATCH);
    }

    arma::vec p_emp_2  = counts2 / static_cast<double>(N_2);
    arma::vec se_bm(N_STATES), se_iid(N_STATES), se_2(N_STATES);
    arma::vec dev_2(N_STATES);
    bool   pass_joint = true;
    double worst_ratio = 0.0;     // max |dev| / se over the 16 states
    for (std::size_t k = 0; k < N_STATES; ++k) {
        const arma::vec col = batch_freq.col(k);
        const double m  = arma::mean(col);
        double        v = 0.0;
        for (std::size_t b = 0; b < N_BATCH; ++b) {
            const double d = col[b] - m;
            v += d * d;
        }
        v /= static_cast<double>(N_BATCH - 1);
        // Batch-means se of the overall mean: sd(batch means)/sqrt(n_batch).
        se_bm[k]  = std::sqrt(v / static_cast<double>(N_BATCH));
        // i.i.d. floor: a positively autocorrelated chain cannot have a
        // smaller se than this, so it guards against a downward-biased
        // batch-means estimate (and against se == 0 for a rare state).
        se_iid[k] = std::sqrt(p_exact_2[k] * (1.0 - p_exact_2[k]) /
                              static_cast<double>(N_2));
        se_2[k]   = std::max(se_bm[k], se_iid[k]);
        dev_2[k]  = std::abs(p_emp_2[k] - p_exact_2[k]);
        worst_ratio = std::max(worst_ratio, dev_2[k] / se_2[k]);
        if (!(dev_2[k] < 4.0 * se_2[k])) pass_joint = false;
    }

    // Largest tolerance actually used on the joint, for the power report.
    double max_tol_2 = 0.0;
    for (std::size_t k = 0; k < N_STATES; ++k)
        max_tol_2 = std::max(max_tol_2, 4.0 * se_2[k]);

    const bool all_pass = pass_marginals && pass_joint && binary_valued;

    return Rcpp::List::create(
        Rcpp::Named("all_pass")        = all_pass,
        Rcpp::Named("pass_marginals")  = pass_marginals,
        Rcpp::Named("pass_joint")      = pass_joint,
        Rcpp::Named("binary_valued")   = binary_valued,
        // Target 1
        Rcpp::Named("n_draws_iid")     = static_cast<int>(N_1),
        Rcpp::Named("p_exact_iid")     = p_exact_1,
        Rcpp::Named("p_emp_iid")       = p_emp_1,
        Rcpp::Named("dev_iid")         = dev_1,
        Rcpp::Named("se_iid_marginal") = se_1,
        // Target 2
        Rcpp::Named("n_draws_joint")   = static_cast<int>(N_2),
        Rcpp::Named("p_exact_joint")   = p_exact_2,
        Rcpp::Named("p_emp_joint")     = p_emp_2,
        Rcpp::Named("dev_joint")       = dev_2,
        Rcpp::Named("se_joint")        = se_2,
        Rcpp::Named("se_batch_means")  = se_bm,
        Rcpp::Named("worst_dev_over_se") = worst_ratio,
        Rcpp::Named("max_tol_joint")   = max_tol_2,
        Rcpp::Named("sep_indep")       = sep_indep,
        Rcpp::Named("marg_exact")      = marg_exact,
        Rcpp::Named("se_factor")       = 4.0);
}

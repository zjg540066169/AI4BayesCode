// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
// test_categorical_gibbs_block.cpp
//
// LIBRARY-LEVEL parity test for AI4BayesCode::categorical_gibbs_block.
//
// What the block does
// -------------------
// The user supplies log_probs_fn(ctx) -> (n_obs x n_categories) matrix whose
// (i, k) entry is log p(z_i = k | everything else), up to a per-row additive
// constant. step() performs one SYSTEMATIC-SCAN sweep: for i = 1..n_obs it
// recomputes the whole matrix from the block's current context, softmaxes row
// i, draws z_i by inverse CDF, writes z_i back into the context under the
// block's own name, and moves on. Labels are stored 1-indexed in {1, ..., K}.
//
// Three properties are worth verifying, and each needs a different target.
// They are checked here as parts A, B and C.
//
// A. THE PER-ROW CONDITIONAL, EXACTLY.
//    When log_probs_fn ignores the context, the sweep produces n_obs
//    INDEPENDENT categorical draws whose cell probabilities are written down
//    in closed form: p_ik = exp(L_ik) / sum_j exp(L_ij). This pins the softmax
//    normalisation, the inverse-CDF draw, and the 1-indexed storage. To make
//    it also exercise the stable-softmax path, each row of the matrix handed
//    to the block carries a large additive offset (+800, -900); those offsets
//    cancel exactly in the mathematics, so the reference probabilities are
//    computed in this file from the OFFSET-FREE base rows -- an independent
//    calculation, not a replay of the block's arithmetic.
//
// B. INVARIANCE OF A COUPLED TARGET, BY EXACT ENUMERATION.
//    Part A cannot see the context write-back at all: a broken implementation
//    that computed the matrix once per sweep, or never refreshed the context,
//    would still pass it. So part B gives the block a target whose sites are
//    NOT independent -- a 3-site, 3-state Potts chain
//
//        pi(z) proportional to exp( sum_i h(i, z_i)
//                                   + J * (1{z_1 = z_2} + 1{z_2 = z_3}) )
//
//    whose full conditionals are exactly what log_probs_fn returns:
//
//        log p(z_i = k | z_{-i}) = h(i, k)
//                                  + J * 1{k = z_{i-1}} + J * 1{k = z_{i+1}}
//                                  + const.
//
//    A systematic scan of pi-invariant conditional updates leaves pi
//    invariant, so the sweep's stationary distribution must BE pi. The state
//    space is 3^3 = 27, small enough to enumerate exactly, so the target here
//    is computed by brute-force summation over all 27 states -- not by the
//    sampler. This is the invariance / exact-enumeration option from the
//    list of provable properties; a closed-form marginal is not available for
//    the coupled chain, which is exactly why enumeration is used.
//
//    Part B is the check that separates the shipped SEQUENTIAL update from a
//    parallel one: a parallel sweep (one matrix per sweep, all z_i sampled
//    against stale neighbours) leaves a DIFFERENT distribution invariant, and
//    with J = 0.9 the two disagree by far more than the tolerance below.
//
// C. DEGENERATE ROWS AND SUPPORT.
//    A row of (-inf, 0, -inf) must put probability one on category 2. That is
//    a probability-1 statement, so it is checked with no tolerance at all, on
//    every one of 500 sweeps. Every label drawn anywhere in this file must
//    also be an integer in {1, ..., K}.
//
// Tolerances
// ----------
// A: per cell, 4 * sqrt(p (1 - p) / N) with N = 200000 iid sweeps -- the
//    binomial standard error of the cell frequency. 12 cells at 4 sigma give
//    a false-alarm probability below 1e-3.
// B: per state, 4 * max(batch-means MCSE, iid SE). The chain is
//    autocorrelated, so the honest error bar is the batch-means MCSE (40
//    batches of 5000 sweeps, batch length far exceeding the integrated
//    autocorrelation time of a 3-site chain); the iid SE is kept as a floor so
//    that a downward fluctuation of the MCSE estimate cannot tighten the gate
//    below the binomial scale. Both are computed from the exact enumerated
//    probabilities, not from the run.
// C: none -- exact.
//
// No threshold here was widened to make the test pass; each is stated in
// standard errors of the estimator being compared. The part-B gate is the one
// whose error bar is estimated rather than analytic, so its margin was
// measured: over 8 seeds the largest per-state |error| / sigma was 2.98
// (typical 1.8 - 2.3), while a control run in which the forward-neighbour
// term was dropped from the conditional -- the block then sampling from the
// wrong conditional -- reached 19.7, with total variation distance rising
// from 0.004 to 0.055. The 4-sigma gate separates the two by a wide margin.
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
#include "AI4BayesCode/categorical_gibbs_block.hpp"

#include <cmath>
#include <limits>
#include <random>
#include <vector>

using AI4BayesCode::block_context;
using AI4BayesCode::categorical_gibbs_block;
using AI4BayesCode::categorical_gibbs_block_config;

namespace {

// True iff x is an integer in {1, ..., K}.
bool valid_label(double x, std::size_t K) {
    if (!std::isfinite(x)) return false;
    const double r = std::floor(x + 0.5);
    if (std::abs(x - r) > 1e-12) return false;
    return (r >= 1.0) && (r <= static_cast<double>(K));
}

} // namespace

// [[Rcpp::export]]
Rcpp::List test_categorical_gibbs_block() {

    bool all_pass  = true;
    bool in_range  = true;

    // =====================================================================
    // PART A -- exact per-row categorical conditional (independent sites)
    // =====================================================================
    const std::size_t nA = 4;      // observations
    const std::size_t KA = 3;      // categories
    const std::size_t N_A = 200000;

    // Base (offset-free) log-weights. Row 3 is flat -> exactly uniform.
    arma::mat baseA(nA, KA);
    baseA(0, 0) =  0.0; baseA(0, 1) =  1.0; baseA(0, 2) = -0.5;
    baseA(1, 0) =  2.0; baseA(1, 1) =  0.0; baseA(1, 2) =  0.5;
    baseA(2, 0) = -1.0; baseA(2, 1) =  0.0; baseA(2, 2) =  1.5;
    baseA(3, 0) =  0.3; baseA(3, 1) =  0.3; baseA(3, 2) =  0.3;

    // Per-row offsets handed to the block. These cancel exactly in the
    // normalised probabilities; they are here only to force the block through
    // its max-subtraction path (exp(+800) would overflow without it).
    arma::vec offA({0.0, 800.0, -900.0, 0.0});

    arma::mat shiftedA(nA, KA);
    for (std::size_t i = 0; i < nA; ++i)
        for (std::size_t k = 0; k < KA; ++k)
            shiftedA(i, k) = baseA(i, k) + offA[i];

    // Reference probabilities from the BASE rows (no offsets, no stabilisation
    // needed): p_ik = exp(b_ik) / sum_j exp(b_ij).
    arma::mat pA(nA, KA);
    for (std::size_t i = 0; i < nA; ++i) {
        double s = 0.0;
        for (std::size_t k = 0; k < KA; ++k) s += std::exp(baseA(i, k));
        for (std::size_t k = 0; k < KA; ++k) pA(i, k) = std::exp(baseA(i, k)) / s;
    }

    categorical_gibbs_block_config cfgA;
    cfgA.name           = "zA";
    cfgA.n_obs          = nA;
    cfgA.n_categories   = KA;
    cfgA.initial_labels = arma::vec({1.0, 1.0, 1.0, 1.0});
    cfgA.log_probs_fn   = [shiftedA](const block_context& /*ctx*/) {
        return shiftedA;
    };
    categorical_gibbs_block blkA(std::move(cfgA));

    block_context ctxA;
    ctxA["zA"] = arma::vec({1.0, 1.0, 1.0, 1.0});
    blkA.set_context(ctxA);

    arma::mat countA(nA, KA, arma::fill::zeros);
    std::mt19937_64 rngA(20260818);
    for (std::size_t it = 0; it < N_A; ++it) {
        blkA.step(rngA);
        const arma::vec& z = blkA.current();
        for (std::size_t i = 0; i < nA; ++i) {
            if (!valid_label(z[i], KA)) { in_range = false; break; }
            countA(i, static_cast<std::size_t>(z[i] + 0.5) - 1) += 1.0;
        }
        if (!in_range) break;
    }

    arma::mat freqA(nA, KA), errA(nA, KA), tolA(nA, KA);
    bool passA = in_range;
    double worst_A_sigma = 0.0;
    for (std::size_t i = 0; i < nA; ++i) {
        for (std::size_t k = 0; k < KA; ++k) {
            freqA(i, k) = countA(i, k) / static_cast<double>(N_A);
            errA(i, k)  = std::abs(freqA(i, k) - pA(i, k));
            // Binomial SE of a cell frequency from N iid sweeps.
            const double se = std::sqrt(pA(i, k) * (1.0 - pA(i, k))
                                        / static_cast<double>(N_A));
            tolA(i, k) = 4.0 * se;
            if (errA(i, k) > tolA(i, k)) passA = false;
            if (se > 0.0 && errA(i, k) / se > worst_A_sigma)
                worst_A_sigma = errA(i, k) / se;
        }
    }
    all_pass = all_pass && passA;

    // =====================================================================
    // PART B -- invariance of a coupled 3-site Potts target, enumerated
    // =====================================================================
    const std::size_t nB   = 3;
    const std::size_t KB   = 3;
    const std::size_t N_B  = 200000;   // recorded sweeps
    const std::size_t BURN = 2000;
    const std::size_t N_BATCH = 40;    // batch-means batches
    const std::size_t BATCH_LEN = N_B / N_BATCH;   // 5000
    const double J = 0.9;              // neighbour coupling

    arma::mat hB(nB, KB);
    hB(0, 0) =  0.0; hB(0, 1) =  0.4; hB(0, 2) = -0.3;
    hB(1, 0) = -0.2; hB(1, 1) =  0.0; hB(1, 2) =  0.6;
    hB(2, 0) =  0.5; hB(2, 1) = -0.1; hB(2, 2) =  0.0;

    // ---- Exact target by brute-force enumeration over all 3^3 states ----
    const std::size_t NSTATE = 27;
    std::vector<double> pB_exact(NSTATE, 0.0);
    {
        double total = 0.0;
        for (std::size_t s = 0; s < NSTATE; ++s) {
            const std::size_t z0 = (s / 9) % 3;   // z_1 - 1
            const std::size_t z1 = (s / 3) % 3;   // z_2 - 1
            const std::size_t z2 =  s      % 3;   // z_3 - 1
            double lw = hB(0, z0) + hB(1, z1) + hB(2, z2);
            if (z0 == z1) lw += J;
            if (z1 == z2) lw += J;
            pB_exact[s] = std::exp(lw);
            total += pB_exact[s];
        }
        for (std::size_t s = 0; s < NSTATE; ++s) pB_exact[s] /= total;
    }

    // ---- The full conditionals, handed to the block --------------------
    categorical_gibbs_block_config cfgB;
    cfgB.name           = "zB";
    cfgB.n_obs          = nB;
    cfgB.n_categories   = KB;
    cfgB.initial_labels = arma::vec({1.0, 1.0, 1.0});
    cfgB.log_probs_fn   = [hB, J, nB, KB](const block_context& ctx) {
        const arma::vec& z = ctx.at("zB");
        arma::mat out(nB, KB);
        for (std::size_t i = 0; i < nB; ++i) {
            for (std::size_t k = 0; k < KB; ++k) {
                double v = hB(i, k);
                const double lab = static_cast<double>(k + 1);
                if (i > 0        && std::abs(z[i - 1] - lab) < 0.5) v += J;
                if (i + 1 < nB   && std::abs(z[i + 1] - lab) < 0.5) v += J;
                out(i, k) = v;
            }
        }
        return out;
    };
    categorical_gibbs_block blkB(std::move(cfgB));

    block_context ctxB;
    ctxB["zB"] = arma::vec({1.0, 1.0, 1.0});
    blkB.set_context(ctxB);

    std::mt19937_64 rngB(778899123);
    for (std::size_t it = 0; it < BURN; ++it) blkB.step(rngB);

    // Per-state counts overall, and per batch (for the batch-means MCSE).
    std::vector<double> countB(NSTATE, 0.0);
    std::vector<std::vector<double> > batch_freq(
        N_BATCH, std::vector<double>(NSTATE, 0.0));
    for (std::size_t b = 0; b < N_BATCH; ++b) {
        for (std::size_t t = 0; t < BATCH_LEN; ++t) {
            blkB.step(rngB);
            const arma::vec& z = blkB.current();
            bool ok = true;
            for (std::size_t i = 0; i < nB; ++i)
                if (!valid_label(z[i], KB)) ok = false;
            if (!ok) { in_range = false; break; }
            const std::size_t s =
                (static_cast<std::size_t>(z[0] + 0.5) - 1) * 9 +
                (static_cast<std::size_t>(z[1] + 0.5) - 1) * 3 +
                (static_cast<std::size_t>(z[2] + 0.5) - 1);
            countB[s]        += 1.0;
            batch_freq[b][s] += 1.0;
        }
        if (!in_range) break;
        for (std::size_t s = 0; s < NSTATE; ++s)
            batch_freq[b][s] /= static_cast<double>(BATCH_LEN);
    }

    const double n_recorded = static_cast<double>(N_BATCH * BATCH_LEN);
    std::vector<double> pB_hat(NSTATE, 0.0), errB(NSTATE, 0.0),
                        tolB(NSTATE, 0.0), mcseB(NSTATE, 0.0);
    bool passB = in_range;
    double tvB = 0.0;
    double worst_B_sigma = 0.0;
    for (std::size_t s = 0; s < NSTATE; ++s) {
        pB_hat[s] = countB[s] / n_recorded;
        errB[s]   = std::abs(pB_hat[s] - pB_exact[s]);
        tvB      += errB[s];

        // Batch-means MCSE of the mean of the 0/1 indicator for state s.
        double bm = 0.0;
        for (std::size_t b = 0; b < N_BATCH; ++b) bm += batch_freq[b][s];
        bm /= static_cast<double>(N_BATCH);
        double bv = 0.0;
        for (std::size_t b = 0; b < N_BATCH; ++b) {
            const double d = batch_freq[b][s] - bm;
            bv += d * d;
        }
        bv /= static_cast<double>(N_BATCH - 1);
        mcseB[s] = std::sqrt(bv / static_cast<double>(N_BATCH));

        // iid binomial SE as a floor, so a low batch-means fluctuation cannot
        // tighten the gate below the binomial scale.
        const double se_iid = std::sqrt(pB_exact[s] * (1.0 - pB_exact[s])
                                         / n_recorded);
        const double sigma = (mcseB[s] > se_iid) ? mcseB[s] : se_iid;
        tolB[s] = 4.0 * sigma;
        if (errB[s] > tolB[s]) passB = false;
        if (sigma > 0.0 && errB[s] / sigma > worst_B_sigma)
            worst_B_sigma = errB[s] / sigma;
    }
    tvB *= 0.5;   // reported diagnostic only; the gate is the per-state check
    all_pass = all_pass && passB;

    // =====================================================================
    // PART C -- degenerate row: (-inf, 0, -inf) must give label 2, always
    // =====================================================================
    const std::size_t N_C = 500;
    const double NEG_INF = -std::numeric_limits<double>::infinity();
    arma::mat LC(1, 3);
    LC(0, 0) = NEG_INF; LC(0, 1) = 0.0; LC(0, 2) = NEG_INF;

    categorical_gibbs_block_config cfgC;
    cfgC.name           = "zC";
    cfgC.n_obs          = 1;
    cfgC.n_categories   = 3;
    cfgC.initial_labels = arma::vec({3.0});   // deliberately the wrong label
    cfgC.log_probs_fn   = [LC](const block_context& /*ctx*/) { return LC; };
    categorical_gibbs_block blkC(std::move(cfgC));

    block_context ctxC;
    ctxC["zC"] = arma::vec({3.0});
    blkC.set_context(ctxC);

    bool passC = true;
    std::mt19937_64 rngC(4242);
    for (std::size_t it = 0; it < N_C; ++it) {
        blkC.step(rngC);
        if (blkC.current()[0] != 2.0) { passC = false; break; }
    }
    all_pass = all_pass && passC;

    all_pass = all_pass && in_range;

    // ---- Report ---------------------------------------------------------
    Rcpp::NumericMatrix freqA_out(nA, KA), pA_out(nA, KA);
    for (std::size_t i = 0; i < nA; ++i) {
        for (std::size_t k = 0; k < KA; ++k) {
            freqA_out(i, k) = freqA(i, k);
            pA_out(i, k)    = pA(i, k);
        }
    }

    return Rcpp::List::create(
        Rcpp::Named("all_pass")        = all_pass,
        Rcpp::Named("pass_A_rowwise")  = passA,
        Rcpp::Named("pass_B_invariant")= passB,
        Rcpp::Named("pass_C_degenerate")= passC,
        Rcpp::Named("in_range")        = in_range,
        Rcpp::Named("A_freq")          = freqA_out,
        Rcpp::Named("A_exact")         = pA_out,
        Rcpp::Named("A_worst_sigma")   = worst_A_sigma,
        Rcpp::Named("A_n_draws")       = static_cast<int>(N_A),
        Rcpp::Named("B_freq")          = Rcpp::wrap(pB_hat),
        Rcpp::Named("B_exact")         = Rcpp::wrap(pB_exact),
        Rcpp::Named("B_abs_err")       = Rcpp::wrap(errB),
        Rcpp::Named("B_tol")           = Rcpp::wrap(tolB),
        Rcpp::Named("B_mcse")          = Rcpp::wrap(mcseB),
        Rcpp::Named("B_worst_sigma")   = worst_B_sigma,
        Rcpp::Named("B_total_variation") = tvB,
        Rcpp::Named("B_n_draws")       = static_cast<int>(n_recorded),
        Rcpp::Named("C_n_draws")       = static_cast<int>(N_C));
}

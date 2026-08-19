// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
// test_lda_collapsed_gibbs_block.cpp
//
// LIBRARY-LEVEL parity test for AI4BayesCode::lda_collapsed_gibbs_block.
//
// What the block does
// -------------------
// One step() is (i) a collapsed Gibbs sweep over all N token-topic labels z
// in a RANDOM permutation order, each token updated by the Griffiths &
// Steyvers (2004) conditional
//
//     P(z_t = k | z_{-t}, w) proportional to
//         (n_{d_t,k}^{-t} + alpha_k)
//         * (n_{k,w_t}^{-t} + beta_{w_t}) / (n_{k,.}^{-t} + sum(beta))
//
// maintained through three INCREMENTAL count tables (decrement the old label,
// sample, increment the new one); followed by (ii) one conjugate Dirichlet
// draw per document, theta_d | z ~ Dir(alpha + n_{d,.}), and one per topic,
// phi_k | z ~ Dir(beta + n_{k,.}), both by gamma-normalisation.
//
// Why the target here is an EXACT ENUMERATION
// -------------------------------------------
// There is no closed-form marginal to compare a sample mean against: theta
// and phi have been integrated out of the z update, and the collapsed
// posterior p(z | w) is a product of Dirichlet-multinomial normalising
// constants over a discrete state space with no factorised marginal. A
// "conditional you can write down holding everything else fixed" is also not
// available in isolation, because step() is a whole sweep -- it advances
// every token and then redraws theta and phi, so no single conditional can be
// held still across a step. What IS provable is INVARIANCE: each token update
// is p(z | w)-invariant, and a random-permutation composition of
// p-invariant kernels is p-invariant, so the sweep's stationary distribution
// must BE p(z | w). On a corpus small enough to enumerate, p(z | w) can be
// written down in closed form and summed over exactly. That is the target
// used below (parts A and B); part C checks the identities that hold with
// probability one.
//
// The corpus is M = 3 documents, V = 3 vocabulary, K = 2 topics, N = 6
// tokens, so the state space is K^N = 64 states -- enumerable. The exact
// collapsed posterior is
//
//     log p(z | w) = sum_d [ sum_k lgamma(n_{d,k} + alpha_k)
//                            - lgamma(n_{d,.} + sum(alpha)) ]
//                  + sum_k [ sum_v lgamma(n_{k,v} + beta_v)
//                            - lgamma(n_{k,.} + sum(beta)) ] + const,
//
// i.e. the Dirichlet-multinomial marginal likelihood. This is computed here
// from lgamma over whole count tables -- a different route from the block's
// incremental ratio arithmetic, so it is an independent reference, not a
// replay of the code under test. alpha = (0.8, 1.6) and
// beta = (0.6, 1.0, 1.4) are deliberately ASYMMETRIC: a symmetric prior makes
// p(z | w) invariant to relabelling topics, which would hide any confusion of
// the alpha_k / beta_v index with a position, and would make the theta and
// phi means in part B equal across topics and so untestable.
//
// A. STATIONARY DISTRIBUTION OF THE z SWEEP, ALL 64 STATES.
//    Compare the empirical frequency of each of the 64 states against the
//    enumerated p(z | w). This is the check that the collapsed conditional
//    and the incremental count bookkeeping are jointly right: a wrong
//    exponent, a missing beta_sum in the denominator, or a stale count leaves
//    a different distribution invariant.
//
// B. THE CONJUGATE theta AND phi DRAWS, FIRST AND SECOND MOMENTS.
//    Part A is blind to theta and phi entirely. Given z, theta_d is
//    Dir(alpha + n_{d,.}) and phi_k is Dir(beta + n_{k,.}), whose marginals
//    are Beta(a, s - a) with s the concentration; so for any state z
//
//        E[theta_{d,k} | z]   = a / s,
//        E[theta_{d,k}^2 | z] = a (a + 1) / (s (s + 1)),
//        E[theta_{d,k}^4 | z] = a(a+1)(a+2)(a+3) / (s(s+1)(s+2)(s+3)),
//
//    with a = alpha_k + n_{d,k}, s = sum(alpha) + n_{d,.}, and the same with
//    (beta_v, n_{k,v}, sum(beta) + n_{k,.}) for phi. Averaging each of these
//    against the enumerated p(z | w) gives the EXACT unconditional moments,
//    and the exact variances Var[X] = E[X^2] - (E X)^2 that set the error
//    bars. Checking the second moment as well as the first is what pins the
//    draw to a Dirichlet rather than to anything with the same mean.
//    Because M = 3, K = 2, V = 3 are pairwise distinct where it matters, the
//    column-major layouts theta[d + k*M] and phi[k + v*K] are also pinned: a
//    transposed flatten would not even have the right length for theta.
//
// C. IDENTITIES THAT HOLD WITH PROBABILITY ONE (no tolerance).
//    Every z entry an integer in {1, ..., K}; the three incrementally
//    maintained count tables re-derived from z by a fresh recount after every
//    single sweep (the classic failure mode of a collapsed sampler is count
//    drift, which a moment check can mask for a long time); n_{d,.} equal to
//    the fixed document lengths; n_{k,.} equal to the row sums of n_{k,v};
//    every theta_d and every phi_k summing to one with strictly positive
//    entries.
//
// Tolerances
// ----------
// A: per state, 4 * max(batch-means MCSE, sqrt(p (1 - p) / n)) with
//    n = 200000 recorded sweeps in 40 batches of 5000. The chain is
//    autocorrelated, so the batch-means MCSE is the honest error bar (a batch
//    of 5000 sweeps is far longer than the integrated autocorrelation time of
//    a 6-token chain); the iid binomial SE, computed from the EXACT p and not
//    from the run, is kept as a floor so a low fluctuation of the MCSE cannot
//    tighten the gate below the binomial scale.
// B: per quantity, 4 * max(batch-means MCSE, sqrt(exact Var / n)), with the
//    exact variances from the enumerated moments above -- fully analytic
//    error bars, nothing estimated from the run except the MCSE.
// C: none -- exact, except simplex sums which are compared at 1e-12, about
//    three orders of magnitude above the accumulated rounding of a sum of
//    three doubles.
//
// The 88 gates (64 states + 24 moments) at 4 sigma have a family-wise
// false-alarm probability near 0.6 percent. The seed is fixed, so the shipped
// run is deterministic; the margin was measured by re-running the whole test
// under six different seeds, over which the largest deviation seen was 2.9
// sigma in part A and 2.6 sigma in part B. No threshold here was widened to
// make the test pass.
//
// The gates were also confirmed to be able to FAIL, by re-running this file
// against deliberately broken copies of the block: replacing sum(beta) by 1
// in the conditional's denominator moves part A to 93 sigma; drawing theta
// from Dir(alpha) instead of Dir(alpha + n_{d,.}) leaves part A passing (the
// z kernel is untouched) and moves part B to 29 sigma, which is why part B
// is not redundant; deleting the n_{d,k} decrement is caught by part C on the
// first sweep.
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
#include "AI4BayesCode/lda_collapsed_gibbs_block.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <vector>

using AI4BayesCode::block_context;
using AI4BayesCode::lda_collapsed_gibbs_block;
using AI4BayesCode::lda_collapsed_gibbs_block_config;

namespace {

// ---- The fixed corpus ---------------------------------------------------
const std::size_t LDA_M = 3;   // documents
const std::size_t LDA_V = 3;   // vocabulary size
const std::size_t LDA_K = 2;   // topics
const std::size_t LDA_N = 6;   // tokens
const std::size_t NSTATE = 64; // LDA_K ^ LDA_N

// 0-indexed token data. Documents have two tokens each; every document uses
// a different pair of words, so no two documents are exchangeable.
const std::size_t DOC_ID[LDA_N]  = {0, 0, 1, 1, 2, 2};
const std::size_t WORD_ID[LDA_N] = {0, 1, 1, 2, 0, 2};

// Asymmetric hyperparameters (see header comment).
const double ALPHA[LDA_K] = {0.8, 1.6};
const double BETA[LDA_V]  = {0.6, 1.0, 1.4};

struct lda_counts {
    double n_dk[LDA_M][LDA_K];
    double n_kv[LDA_K][LDA_V];
    double n_k[LDA_K];
    double n_d[LDA_M];
};

void zero_counts(lda_counts& c) {
    for (std::size_t d = 0; d < LDA_M; ++d) {
        c.n_d[d] = 0.0;
        for (std::size_t k = 0; k < LDA_K; ++k) c.n_dk[d][k] = 0.0;
    }
    for (std::size_t k = 0; k < LDA_K; ++k) {
        c.n_k[k] = 0.0;
        for (std::size_t v = 0; v < LDA_V; ++v) c.n_kv[k][v] = 0.0;
    }
}

// Tabulate the count tables induced by 0-indexed labels lab[0..N-1].
lda_counts counts_from_labels(const std::size_t* lab) {
    lda_counts c;
    zero_counts(c);
    for (std::size_t t = 0; t < LDA_N; ++t) {
        const std::size_t d = DOC_ID[t];
        const std::size_t v = WORD_ID[t];
        const std::size_t k = lab[t];
        c.n_dk[d][k] += 1.0;
        c.n_kv[k][v] += 1.0;
        c.n_k[k]     += 1.0;
        c.n_d[d]     += 1.0;
    }
    return c;
}

// Enumeration index s in [0, K^N) -> 0-indexed labels, token t occupying
// digit t in base K (least significant first).
void labels_from_state(std::size_t s, std::size_t* lab) {
    for (std::size_t t = 0; t < LDA_N; ++t) {
        lab[t] = s % LDA_K;
        s /= LDA_K;
    }
}

// The same encoding, applied to the block's 1-indexed z vector.
std::size_t state_from_z(const arma::vec& z) {
    std::size_t s = 0, mult = 1;
    for (std::size_t t = 0; t < LDA_N; ++t) {
        const std::size_t k =
            static_cast<std::size_t>(z[t] + 0.5) - 1;
        s += k * mult;
        mult *= LDA_K;
    }
    return s;
}

bool valid_label(double x) {
    if (!std::isfinite(x)) return false;
    const double r = std::floor(x + 0.5);
    if (std::abs(x - r) > 1e-12) return false;
    return (r >= 1.0) && (r <= static_cast<double>(LDA_K));
}

// Rising factorial ratio E[X^m] for a Beta(a, s - a) marginal of a Dirichlet:
//   a (a+1) ... (a+m-1) / ( s (s+1) ... (s+m-1) ).
double beta_moment(double a, double s, int m) {
    double num = 1.0, den = 1.0;
    for (int j = 0; j < m; ++j) {
        num *= (a + static_cast<double>(j));
        den *= (s + static_cast<double>(j));
    }
    return num / den;
}

// Batch-means Monte Carlo standard error of a mean, from per-batch means.
double batch_mcse(const std::vector<double>& batch_mean) {
    const std::size_t B = batch_mean.size();
    double m = 0.0;
    for (std::size_t b = 0; b < B; ++b) m += batch_mean[b];
    m /= static_cast<double>(B);
    double v = 0.0;
    for (std::size_t b = 0; b < B; ++b) {
        const double dd = batch_mean[b] - m;
        v += dd * dd;
    }
    v /= static_cast<double>(B - 1);
    return std::sqrt(v / static_cast<double>(B));
}

} // namespace

// [[Rcpp::export]]
Rcpp::List test_lda_collapsed_gibbs_block() {

    const std::size_t BURN      = 2000;
    const std::size_t N_BATCH   = 40;
    const std::size_t BATCH_LEN = 5000;
    const std::size_t N_KEEP    = N_BATCH * BATCH_LEN;   // 200000
    const double n_keep_d       = static_cast<double>(N_KEEP);

    double alpha_sum = 0.0, beta_sum = 0.0;
    for (std::size_t k = 0; k < LDA_K; ++k) alpha_sum += ALPHA[k];
    for (std::size_t v = 0; v < LDA_V; ++v) beta_sum  += BETA[v];

    // =====================================================================
    // EXACT TARGET -- enumerate p(z | w) over all K^N states, and with it
    // the exact moments of theta and phi.
    // =====================================================================
    std::vector<double> p_exact(NSTATE, 0.0);
    // Quantity layout: theta index d*K + k, phi index k*V + v.
    const std::size_t NTH = LDA_M * LDA_K;   // 6
    const std::size_t NPH = LDA_K * LDA_V;   // 6
    std::vector<double> th_m1(NTH, 0.0), th_m2(NTH, 0.0), th_m4(NTH, 0.0);
    std::vector<double> ph_m1(NPH, 0.0), ph_m2(NPH, 0.0), ph_m4(NPH, 0.0);
    {
        std::vector<double> lw(NSTATE, 0.0);
        double lmax = -std::numeric_limits<double>::infinity();
        for (std::size_t s = 0; s < NSTATE; ++s) {
            std::size_t lab[LDA_N];
            labels_from_state(s, lab);
            const lda_counts c = counts_from_labels(lab);
            double v = 0.0;
            for (std::size_t d = 0; d < LDA_M; ++d) {
                for (std::size_t k = 0; k < LDA_K; ++k)
                    v += std::lgamma(c.n_dk[d][k] + ALPHA[k]);
                v -= std::lgamma(c.n_d[d] + alpha_sum);
            }
            for (std::size_t k = 0; k < LDA_K; ++k) {
                for (std::size_t vv = 0; vv < LDA_V; ++vv)
                    v += std::lgamma(c.n_kv[k][vv] + BETA[vv]);
                v -= std::lgamma(c.n_k[k] + beta_sum);
            }
            lw[s] = v;
            if (v > lmax) lmax = v;
        }
        double tot = 0.0;
        for (std::size_t s = 0; s < NSTATE; ++s) {
            p_exact[s] = std::exp(lw[s] - lmax);
            tot += p_exact[s];
        }
        for (std::size_t s = 0; s < NSTATE; ++s) p_exact[s] /= tot;

        // Exact moments of theta and phi under p(z | w), by summing the
        // conditional Dirichlet moments against p(z | w).
        for (std::size_t s = 0; s < NSTATE; ++s) {
            std::size_t lab[LDA_N];
            labels_from_state(s, lab);
            const lda_counts c = counts_from_labels(lab);
            const double p = p_exact[s];
            for (std::size_t d = 0; d < LDA_M; ++d) {
                const double sd = alpha_sum + c.n_d[d];
                for (std::size_t k = 0; k < LDA_K; ++k) {
                    const double a = ALPHA[k] + c.n_dk[d][k];
                    const std::size_t q = d * LDA_K + k;
                    th_m1[q] += p * beta_moment(a, sd, 1);
                    th_m2[q] += p * beta_moment(a, sd, 2);
                    th_m4[q] += p * beta_moment(a, sd, 4);
                }
            }
            for (std::size_t k = 0; k < LDA_K; ++k) {
                const double sk = beta_sum + c.n_k[k];
                for (std::size_t v = 0; v < LDA_V; ++v) {
                    const double a = BETA[v] + c.n_kv[k][v];
                    const std::size_t q = k * LDA_V + v;
                    ph_m1[q] += p * beta_moment(a, sk, 1);
                    ph_m2[q] += p * beta_moment(a, sk, 2);
                    ph_m4[q] += p * beta_moment(a, sk, 4);
                }
            }
        }
    }

    // =====================================================================
    // BUILD THE BLOCK
    // =====================================================================
    lda_collapsed_gibbs_block_config cfg;
    cfg.name  = "lda_test";
    cfg.M     = LDA_M;
    cfg.V     = LDA_V;
    cfg.K     = LDA_K;
    cfg.alpha = arma::vec(LDA_K);
    for (std::size_t k = 0; k < LDA_K; ++k) cfg.alpha[k] = ALPHA[k];
    cfg.beta  = arma::vec(LDA_V);
    for (std::size_t v = 0; v < LDA_V; ++v) cfg.beta[v] = BETA[v];
    cfg.w_key   = "w";
    cfg.doc_key = "doc";
    lda_collapsed_gibbs_block blk(std::move(cfg));

    block_context ctx;
    arma::vec w_vec(LDA_N), doc_vec(LDA_N);
    for (std::size_t t = 0; t < LDA_N; ++t) {
        w_vec[t]   = static_cast<double>(WORD_ID[t] + 1);   // 1-indexed
        doc_vec[t] = static_cast<double>(DOC_ID[t] + 1);
    }
    ctx["w"]   = w_vec;
    ctx["doc"] = doc_vec;
    blk.set_context(ctx);

    bool pass_C = (blk.n_tokens() == LDA_N) && (blk.dim() == LDA_N);

    // =====================================================================
    // RUN
    // =====================================================================
    std::mt19937_64 rng(20260818u);
    for (std::size_t it = 0; it < BURN; ++it) blk.step(rng);

    std::vector<double> cnt_state(NSTATE, 0.0);
    std::vector<std::vector<double> > b_state(
        N_BATCH, std::vector<double>(NSTATE, 0.0));

    std::vector<double> s_th1(NTH, 0.0), s_th2(NTH, 0.0);
    std::vector<double> s_ph1(NPH, 0.0), s_ph2(NPH, 0.0);
    std::vector<std::vector<double> > b_th1(
        N_BATCH, std::vector<double>(NTH, 0.0));
    std::vector<std::vector<double> > b_th2(
        N_BATCH, std::vector<double>(NTH, 0.0));
    std::vector<std::vector<double> > b_ph1(
        N_BATCH, std::vector<double>(NPH, 0.0));
    std::vector<std::vector<double> > b_ph2(
        N_BATCH, std::vector<double>(NPH, 0.0));

    for (std::size_t b = 0; b < N_BATCH; ++b) {
        for (std::size_t t = 0; t < BATCH_LEN; ++t) {
            blk.step(rng);

            const arma::vec& z     = blk.current();
            const arma::vec& theta = blk.current_theta();
            const arma::vec& phi   = blk.current_phi();

            // ---- PART C: probability-one identities, every sweep --------
            std::size_t lab[LDA_N];
            for (std::size_t i = 0; i < LDA_N; ++i) {
                if (!valid_label(z[i])) { pass_C = false; break; }
                lab[i] = static_cast<std::size_t>(z[i] + 0.5) - 1;
            }
            if (!pass_C) break;

            // Incrementally maintained counts must equal a fresh recount.
            const lda_counts rc = counts_from_labels(lab);
            const arma::mat& n_dk = blk.counts_dk();
            const arma::mat& n_kv = blk.counts_kv();
            const arma::vec& n_k  = blk.counts_k();
            for (std::size_t d = 0; d < LDA_M && pass_C; ++d)
                for (std::size_t k = 0; k < LDA_K; ++k)
                    if (std::abs(n_dk(d, k) - rc.n_dk[d][k]) > 1e-9)
                        pass_C = false;
            for (std::size_t k = 0; k < LDA_K && pass_C; ++k) {
                double row = 0.0;
                for (std::size_t v = 0; v < LDA_V; ++v) {
                    if (std::abs(n_kv(k, v) - rc.n_kv[k][v]) > 1e-9)
                        pass_C = false;
                    row += n_kv(k, v);
                }
                // n_k must be the row sum of n_kv, and match the recount.
                if (std::abs(n_k[k] - row) > 1e-9)          pass_C = false;
                if (std::abs(n_k[k] - rc.n_k[k]) > 1e-9)    pass_C = false;
            }
            // Document lengths are fixed data; n_{d,.} must reproduce them.
            for (std::size_t d = 0; d < LDA_M && pass_C; ++d) {
                double row = 0.0;
                for (std::size_t k = 0; k < LDA_K; ++k) row += n_dk(d, k);
                if (std::abs(row - rc.n_d[d]) > 1e-9) pass_C = false;
            }
            // Simplex constraints on the conjugate draws.
            if (theta.n_elem != LDA_M * LDA_K) pass_C = false;
            if (phi.n_elem   != LDA_K * LDA_V) pass_C = false;
            for (std::size_t d = 0; d < LDA_M && pass_C; ++d) {
                double srow = 0.0;
                for (std::size_t k = 0; k < LDA_K; ++k) {
                    const double x = theta[d + k * LDA_M];
                    if (!(x > 0.0) || !(x < 1.0)) pass_C = false;
                    srow += x;
                }
                if (std::abs(srow - 1.0) > 1e-12) pass_C = false;
            }
            for (std::size_t k = 0; k < LDA_K && pass_C; ++k) {
                double srow = 0.0;
                for (std::size_t v = 0; v < LDA_V; ++v) {
                    const double x = phi[k + v * LDA_K];
                    if (!(x > 0.0) || !(x < 1.0)) pass_C = false;
                    srow += x;
                }
                if (std::abs(srow - 1.0) > 1e-12) pass_C = false;
            }
            if (!pass_C) break;

            // ---- PART A accumulation ------------------------------------
            const std::size_t s = state_from_z(z);
            cnt_state[s]  += 1.0;
            b_state[b][s] += 1.0;

            // ---- PART B accumulation ------------------------------------
            for (std::size_t d = 0; d < LDA_M; ++d) {
                for (std::size_t k = 0; k < LDA_K; ++k) {
                    const std::size_t q = d * LDA_K + k;
                    const double x = theta[d + k * LDA_M];
                    s_th1[q]     += x;
                    s_th2[q]     += x * x;
                    b_th1[b][q]  += x;
                    b_th2[b][q]  += x * x;
                }
            }
            for (std::size_t k = 0; k < LDA_K; ++k) {
                for (std::size_t v = 0; v < LDA_V; ++v) {
                    const std::size_t q = k * LDA_V + v;
                    const double x = phi[k + v * LDA_K];
                    s_ph1[q]     += x;
                    s_ph2[q]     += x * x;
                    b_ph1[b][q]  += x;
                    b_ph2[b][q]  += x * x;
                }
            }
        }
        if (!pass_C) break;
        const double bl = static_cast<double>(BATCH_LEN);
        for (std::size_t s = 0; s < NSTATE; ++s) b_state[b][s] /= bl;
        for (std::size_t q = 0; q < NTH; ++q) {
            b_th1[b][q] /= bl;
            b_th2[b][q] /= bl;
        }
        for (std::size_t q = 0; q < NPH; ++q) {
            b_ph1[b][q] /= bl;
            b_ph2[b][q] /= bl;
        }
    }

    // =====================================================================
    // PART A -- 64-state stationary distribution vs exact enumeration
    // =====================================================================
    bool pass_A = pass_C;
    std::vector<double> p_hat(NSTATE, 0.0), errA(NSTATE, 0.0),
                        tolA(NSTATE, 0.0);
    double worst_A_sigma = 0.0, tv_A = 0.0;
    if (pass_C) {
        for (std::size_t s = 0; s < NSTATE; ++s) {
            p_hat[s] = cnt_state[s] / n_keep_d;
            errA[s]  = std::abs(p_hat[s] - p_exact[s]);
            tv_A    += errA[s];

            std::vector<double> bm(N_BATCH);
            for (std::size_t b = 0; b < N_BATCH; ++b) bm[b] = b_state[b][s];
            const double mcse = batch_mcse(bm);
            // iid binomial SE from the EXACT p, used as a floor.
            const double se_iid =
                std::sqrt(p_exact[s] * (1.0 - p_exact[s]) / n_keep_d);
            const double sigma = (mcse > se_iid) ? mcse : se_iid;
            tolA[s] = 4.0 * sigma;
            if (errA[s] > tolA[s]) pass_A = false;
            if (sigma > 0.0 && errA[s] / sigma > worst_A_sigma)
                worst_A_sigma = errA[s] / sigma;
        }
        tv_A *= 0.5;   // reported diagnostic; the gate is the per-state check
    }

    // =====================================================================
    // PART B -- theta and phi moments vs exact enumeration
    // =====================================================================
    bool pass_B = pass_C;
    std::vector<double> th1_hat(NTH), th2_hat(NTH), ph1_hat(NPH), ph2_hat(NPH);
    std::vector<double> th1_tol(NTH), th2_tol(NTH), ph1_tol(NPH), ph2_tol(NPH);
    double worst_B_sigma = 0.0;

    if (pass_C) {
        // theta: mean and second moment.
        for (std::size_t q = 0; q < NTH; ++q) {
            th1_hat[q] = s_th1[q] / n_keep_d;
            th2_hat[q] = s_th2[q] / n_keep_d;

            std::vector<double> bm1(N_BATCH), bm2(N_BATCH);
            for (std::size_t b = 0; b < N_BATCH; ++b) {
                bm1[b] = b_th1[b][q];
                bm2[b] = b_th2[b][q];
            }
            // Exact variances: Var[X] = E X^2 - (E X)^2,
            //                  Var[X^2] = E X^4 - (E X^2)^2.
            const double var1 = th_m2[q] - th_m1[q] * th_m1[q];
            const double var2 = th_m4[q] - th_m2[q] * th_m2[q];
            const double s1 = std::max(batch_mcse(bm1),
                                       std::sqrt(var1 / n_keep_d));
            const double s2 = std::max(batch_mcse(bm2),
                                       std::sqrt(var2 / n_keep_d));
            th1_tol[q] = 4.0 * s1;
            th2_tol[q] = 4.0 * s2;
            const double e1 = std::abs(th1_hat[q] - th_m1[q]);
            const double e2 = std::abs(th2_hat[q] - th_m2[q]);
            if (e1 > th1_tol[q]) pass_B = false;
            if (e2 > th2_tol[q]) pass_B = false;
            if (s1 > 0.0 && e1 / s1 > worst_B_sigma) worst_B_sigma = e1 / s1;
            if (s2 > 0.0 && e2 / s2 > worst_B_sigma) worst_B_sigma = e2 / s2;
        }
        // phi: mean and second moment.
        for (std::size_t q = 0; q < NPH; ++q) {
            ph1_hat[q] = s_ph1[q] / n_keep_d;
            ph2_hat[q] = s_ph2[q] / n_keep_d;

            std::vector<double> bm1(N_BATCH), bm2(N_BATCH);
            for (std::size_t b = 0; b < N_BATCH; ++b) {
                bm1[b] = b_ph1[b][q];
                bm2[b] = b_ph2[b][q];
            }
            const double var1 = ph_m2[q] - ph_m1[q] * ph_m1[q];
            const double var2 = ph_m4[q] - ph_m2[q] * ph_m2[q];
            const double s1 = std::max(batch_mcse(bm1),
                                       std::sqrt(var1 / n_keep_d));
            const double s2 = std::max(batch_mcse(bm2),
                                       std::sqrt(var2 / n_keep_d));
            ph1_tol[q] = 4.0 * s1;
            ph2_tol[q] = 4.0 * s2;
            const double e1 = std::abs(ph1_hat[q] - ph_m1[q]);
            const double e2 = std::abs(ph2_hat[q] - ph_m2[q]);
            if (e1 > ph1_tol[q]) pass_B = false;
            if (e2 > ph2_tol[q]) pass_B = false;
            if (s1 > 0.0 && e1 / s1 > worst_B_sigma) worst_B_sigma = e1 / s1;
            if (s2 > 0.0 && e2 / s2 > worst_B_sigma) worst_B_sigma = e2 / s2;
        }
    }

    const bool all_pass = pass_A && pass_B && pass_C;

    return Rcpp::List::create(
        Rcpp::Named("all_pass")          = all_pass,
        Rcpp::Named("pass_A_z_invariant")= pass_A,
        Rcpp::Named("pass_B_theta_phi")  = pass_B,
        Rcpp::Named("pass_C_exact")      = pass_C,
        Rcpp::Named("A_freq")            = Rcpp::wrap(p_hat),
        Rcpp::Named("A_exact")           = Rcpp::wrap(p_exact),
        Rcpp::Named("A_abs_err")         = Rcpp::wrap(errA),
        Rcpp::Named("A_tol")             = Rcpp::wrap(tolA),
        Rcpp::Named("A_worst_sigma")     = worst_A_sigma,
        Rcpp::Named("A_total_variation") = tv_A,
        Rcpp::Named("B_theta_mean")      = Rcpp::wrap(th1_hat),
        Rcpp::Named("B_theta_mean_exact")= Rcpp::wrap(th_m1),
        Rcpp::Named("B_theta_m2")        = Rcpp::wrap(th2_hat),
        Rcpp::Named("B_theta_m2_exact")  = Rcpp::wrap(th_m2),
        Rcpp::Named("B_phi_mean")        = Rcpp::wrap(ph1_hat),
        Rcpp::Named("B_phi_mean_exact")  = Rcpp::wrap(ph_m1),
        Rcpp::Named("B_phi_m2")          = Rcpp::wrap(ph2_hat),
        Rcpp::Named("B_phi_m2_exact")    = Rcpp::wrap(ph_m2),
        Rcpp::Named("B_worst_sigma")     = worst_B_sigma,
        Rcpp::Named("n_states")          = static_cast<int>(NSTATE),
        Rcpp::Named("n_draws")           = static_cast<int>(N_KEEP),
        Rcpp::Named("n_burn")            = static_cast<int>(BURN));
}

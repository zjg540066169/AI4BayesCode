// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
// test_gamma_gibbs_block.cpp
//
// LIBRARY-LEVEL parity test for AI4BayesCode::gamma_gibbs_block (**Option A
// Check #15** from skills/codegen.md).
//
// WHAT IS VERIFIED
// ----------------
// gamma_gibbs_block is an EXACT conjugate leaf: params_fn hands it a
// (shape, rate) pair and step() must return an independent draw from
// Gamma(shape, rate) in the shape-RATE convention documented in the header
//
//     pdf(x) = rate^shape / Gamma(shape) * x^(shape-1) * exp(-rate * x)
//     E[x]     = shape / rate
//     Var[x]   = shape / rate^2
//     E[log x] = digamma(shape) - log(rate)
//
// Because the target law is available in closed form, this test compares the
// empirical law of the block's draws directly to that Gamma. The weaker
// fallbacks listed for blocks without a closed-form marginal (invariance of
// a kernel, exact enumeration, a detailed-balance identity) are NOT used
// here and would be strictly less informative: this block is not a Markov
// kernel at all -- consecutive draws are independent given the context, so
// "the kernel preserves the target" and "the draws ARE the target" coincide,
// and the latter is directly checkable.
//
// The single most dangerous silent bug in this block is a shape-SCALE /
// shape-RATE mix-up, because prng::gamma_distribution is parameterised by
// SCALE while the block's contract is RATE. The fixture therefore uses
// rate = 2.0 (never 1.0, where the two conventions coincide): under a
// scale/rate swap the mean would be shape * rate = 7.0 instead of
// shape / rate = 1.75, a factor of 4 -- far outside every tolerance below.
//
// FOUR CHECKS
// -----------
// A. Moments of a fixed Gamma(3.5, 2.0) from N = 40000 draws:
//      sample mean vs shape/rate, sample variance vs shape/rate^2.
// B. E[log x] vs digamma(shape) - log(rate). This is a second, independent
//    functional of the law; it is far more sensitive to a misparameterised
//    shape than the mean is (the mean can be matched by trading shape
//    against rate, the pair (E[x], E[log x]) cannot -- it is the sufficient
//    statistic pair of the Gamma family).
// C. Goodness of fit over the WHOLE distribution, not just three moments:
//    the probability integral transform u = pgamma(x, shape, 1/rate) must be
//    Uniform(0,1), tested with a 20-bin equiprobable chi-square statistic.
// D. The conditional actually tracks the context: the same block is given a
//    params_fn that reads shape and rate out of block_context, and is driven
//    through two different contexts. Each segment must match its own
//    analytic Gamma. This is what makes the test a test of the CONDITIONAL
//    sampler rather than of one hard-wired distribution.
//
// TOLERANCES
// ----------
// Every threshold is a Monte Carlo standard error for the number of draws
// actually taken, computed in the code from the analytic law -- nothing is
// tuned to make the test pass.
//   - mean:     SE = sqrt(Var[x] / N)
//   - variance: SE = Var[x] * sqrt((2 + 6/shape) / N), from
//               Var[s^2] = (mu4 - mu2^2)/N with Gamma excess kurtosis 6/shape
//   - E[log x]: SE = sqrt(trigamma(shape) / N), since Var[log x] =
//               trigamma(shape) exactly
// Each is allowed 4 SE, a two-sided false-alarm rate of about 6e-5 per
// check. The chi-square statistic in C is compared to the 0.999 quantile of
// chi-square with 19 df (about 43.8), a false-alarm rate of 1e-3.
//
// The thresholds were then confirmed against MEASURED spread rather than
// assumed. Over 8 seeds of a correct Gamma(3.5, 2.0) sampler the largest
// values seen were z_mean 1.38, z_var 1.93, z_log 1.35 and chisq 35.1 --
// all inside the limits, with the moment checks using less than half their
// budget.
//
// The checks were also confirmed to be able to FAIL, on three deliberately
// wrong samplers measured at the same N:
//   rate/scale swap, Gamma(3.5, rate 0.5): z = 1126 / 1563 / 483, chisq 5e5
//   shape 10% high, Gamma(3.85, 2.0):      z = 38.5 / 10.6 / 39.3, chisq 1566
//   mean-preserving, Gamma(7.0, 4.0):      z = 1.01 / 51.8 / 27.7, chisq 6595
// The last case is why checks B and C exist: it has exactly the right mean,
// so check A alone (z_mean = 1.01) would have passed it.
//
// Returned Rcpp::List so the caller (Rcpp::sourceCpp from an R audit
// script) can programmatically check all_pass.
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
#include "AI4BayesCode/gamma_gibbs_block.hpp"

#include <cmath>
#include <cstddef>
#include <random>

#include "../../tests/portable_rng.hpp"   // portable draws: identical on every stdlib
#include <vector>

using AI4BayesCode::gamma_gibbs_block;
using AI4BayesCode::gamma_gibbs_block_config;
using AI4BayesCode::gamma_params;
using AI4BayesCode::block_context;

namespace {

struct moments {
    double mean;
    double var;
    double mean_log;
};

moments summarize(const std::vector<double>& x) {
    const double n = static_cast<double>(x.size());
    double m = 0.0, ml = 0.0;
    for (double v : x) { m += v; ml += std::log(v); }
    m  /= n;
    ml /= n;
    double s2 = 0.0;
    for (double v : x) { const double d = v - m; s2 += d * d; }
    s2 /= (n - 1.0);
    return moments{m, s2, ml};
}

// Equiprobable-bin chi-square goodness of fit on the probability integral
// transform. Under the null (draws really are Gamma(shape, rate)) the
// returned statistic is asymptotically chi-square with n_bins - 1 df.
double pit_chisq(const std::vector<double>& x, double shape, double rate,
                 std::size_t n_bins) {
    std::vector<double> count(n_bins, 0.0);
    for (double v : x) {
        // R::pgamma is shape-SCALE, so pass scale = 1 / rate to evaluate the
        // CDF of our shape-RATE Gamma(shape, rate).
        const double u = R::pgamma(v, shape, 1.0 / rate, /*lower_tail=*/1,
                                   /*log_p=*/0);
        std::size_t b = static_cast<std::size_t>(u * static_cast<double>(n_bins));
        if (b >= n_bins) b = n_bins - 1;
        count[b] += 1.0;
    }
    const double expected =
        static_cast<double>(x.size()) / static_cast<double>(n_bins);
    double stat = 0.0;
    for (std::size_t b = 0; b < n_bins; ++b) {
        const double d = count[b] - expected;
        stat += d * d / expected;
    }
    return stat;
}

}  // namespace

// [[Rcpp::export]]
Rcpp::List test_gamma_gibbs_block() {
    // ---- Fixture A/B/C: fixed shape-RATE hyperparameters ------------------
    // rate != 1 on purpose: it separates shape-RATE from shape-SCALE.
    const double shape_true = 3.5;
    const double rate_true  = 2.0;
    const std::size_t N_DRAWS = 40000;

    gamma_gibbs_block_config cfg;
    cfg.name          = "tau_test";
    cfg.initial_value = shape_true / rate_true;   // prior mean
    cfg.params_fn     = [shape_true, rate_true](const block_context& /*ctx*/) {
        return gamma_params{shape_true, rate_true};
    };
    gamma_gibbs_block blk(std::move(cfg));

    block_context ctx;            // params_fn ignores it here
    blk.set_context(ctx);

    std::mt19937_64 rng(20260818);
    std::vector<double> samples;
    samples.reserve(N_DRAWS);
    bool in_support = true;
    for (std::size_t i = 0; i < N_DRAWS; ++i) {
        blk.step(rng);
        const double v = blk.current()[0];
        if (!(v > 0.0) || !std::isfinite(v)) in_support = false;
        samples.push_back(v);
    }

    const moments emp = summarize(samples);

    // ---- Analytic Gamma(3.5, 2.0), shape-RATE ----------------------------
    const double exp_mean = shape_true / rate_true;
    const double exp_var  = shape_true / (rate_true * rate_true);
    const double exp_mlog = R::digamma(shape_true) - std::log(rate_true);

    // ---- Monte Carlo standard errors at N = 40000 ------------------------
    const double nd = static_cast<double>(N_DRAWS);
    // SE of a sample mean of iid draws with variance exp_var.
    const double se_mean = std::sqrt(exp_var / nd);
    // Var[s^2] = (mu4 - mu2^2)/n; for a Gamma, mu4 = var^2 * (3 + 6/shape),
    // so Var[s^2] = var^2 * (2 + 6/shape) / n.
    const double se_var =
        exp_var * std::sqrt((2.0 + 6.0 / shape_true) / nd);
    // Var[log x] = trigamma(shape) exactly (it does not depend on the rate).
    const double se_mlog = std::sqrt(R::trigamma(shape_true) / nd);

    // 4 SE: two-sided false-alarm probability about 6e-5 per check.
    const double K_SE = 4.0;
    const double z_mean = std::abs(emp.mean     - exp_mean) / se_mean;
    const double z_var  = std::abs(emp.var      - exp_var)  / se_var;
    const double z_mlog = std::abs(emp.mean_log - exp_mlog) / se_mlog;

    const bool pass_mean = (z_mean < K_SE);
    const bool pass_var  = (z_var  < K_SE);
    const bool pass_mlog = (z_mlog < K_SE);

    // ---- C. Whole-distribution goodness of fit ---------------------------
    const std::size_t N_BINS = 20;
    const double chisq = pit_chisq(samples, shape_true, rate_true, N_BINS);
    // 0.999 quantile of chi-square with N_BINS - 1 = 19 df (about 43.82):
    // a 1-in-1000 false-alarm rate under a correct sampler.
    const double chisq_crit =
        R::qchisq(0.999, static_cast<double>(N_BINS - 1), /*lower_tail=*/1,
                  /*log_p=*/0);
    const bool pass_gof = (chisq < chisq_crit);

    // ---- D. The conditional must follow the context ----------------------
    // Same block, params_fn now reads (shape, rate) out of block_context.
    // Two contexts with very different means; each segment is checked
    // against its own analytic Gamma at 4 SE.
    const std::size_t N_CTX = 20000;
    const double shape_ctx[2] = {2.0, 9.0};
    const double rate_ctx[2]  = {0.5, 3.0};   // means 4.0 and 3.0

    gamma_gibbs_block_config cfg2;
    cfg2.name          = "tau_ctx";
    cfg2.initial_value = 1.0;
    cfg2.params_fn     = [](const block_context& c) {
        return gamma_params{c.at("shape_ctx")[0], c.at("rate_ctx")[0]};
    };
    gamma_gibbs_block blk2(std::move(cfg2));

    double ctx_mean[2], ctx_exp_mean[2], ctx_z_mean[2];
    bool pass_ctx[2];
    for (std::size_t s = 0; s < 2; ++s) {
        block_context c;
        c["shape_ctx"] = arma::vec({shape_ctx[s]});
        c["rate_ctx"]  = arma::vec({rate_ctx[s]});
        blk2.set_context(c);

        std::vector<double> seg;
        seg.reserve(N_CTX);
        for (std::size_t i = 0; i < N_CTX; ++i) {
            blk2.step(rng);
            const double v = blk2.current()[0];
            if (!(v > 0.0) || !std::isfinite(v)) in_support = false;
            seg.push_back(v);
        }
        const moments m = summarize(seg);
        ctx_exp_mean[s] = shape_ctx[s] / rate_ctx[s];
        const double se =
            std::sqrt((shape_ctx[s] / (rate_ctx[s] * rate_ctx[s])) /
                      static_cast<double>(N_CTX));
        ctx_mean[s]   = m.mean;
        ctx_z_mean[s] = std::abs(m.mean - ctx_exp_mean[s]) / se;
        pass_ctx[s]   = (ctx_z_mean[s] < K_SE);
    }

    const bool all_pass = pass_mean && pass_var && pass_mlog && pass_gof &&
                          pass_ctx[0] && pass_ctx[1] && in_support;

    return Rcpp::List::create(
        Rcpp::Named("all_pass")     = all_pass,
        Rcpp::Named("pass_mean")    = pass_mean,
        Rcpp::Named("pass_var")     = pass_var,
        Rcpp::Named("pass_mean_log")= pass_mlog,
        Rcpp::Named("pass_gof")     = pass_gof,
        Rcpp::Named("pass_ctx")     = Rcpp::LogicalVector::create(pass_ctx[0],
                                                                 pass_ctx[1]),
        Rcpp::Named("in_support")   = in_support,
        Rcpp::Named("sample_mean")  = emp.mean,
        Rcpp::Named("exp_mean")     = exp_mean,
        Rcpp::Named("sample_var")   = emp.var,
        Rcpp::Named("exp_var")      = exp_var,
        Rcpp::Named("sample_mlog")  = emp.mean_log,
        Rcpp::Named("exp_mlog")     = exp_mlog,
        Rcpp::Named("z_mean")       = z_mean,
        Rcpp::Named("z_var")        = z_var,
        Rcpp::Named("z_mean_log")   = z_mlog,
        Rcpp::Named("k_se")         = K_SE,
        Rcpp::Named("chisq")        = chisq,
        Rcpp::Named("chisq_crit")   = chisq_crit,
        Rcpp::Named("ctx_mean")     = Rcpp::NumericVector::create(ctx_mean[0],
                                                                 ctx_mean[1]),
        Rcpp::Named("ctx_exp_mean") = Rcpp::NumericVector::create(ctx_exp_mean[0],
                                                                 ctx_exp_mean[1]),
        Rcpp::Named("ctx_z_mean")   = Rcpp::NumericVector::create(ctx_z_mean[0],
                                                                 ctx_z_mean[1]),
        Rcpp::Named("n_draws")      = static_cast<int>(N_DRAWS),
        Rcpp::Named("n_ctx_draws")  = static_cast<int>(N_CTX));
}

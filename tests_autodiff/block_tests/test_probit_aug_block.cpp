// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
// test_probit_aug_block.cpp
//
// LIBRARY-LEVEL parity test for AI4BayesCode::probit_aug_block (Check #15).
//
// WHAT THE BLOCK CLAIMS
// ---------------------
// probit_aug_block performs the Albert-Chib (1993) data-augmentation step of
// a probit binary model. Given y_i in {0,1} and a linear predictor
// mu_i (plus an optional offset), it draws
//
//     z_i | rest ~ N(m_i, 1) truncated to (0, +inf)  if y_i = 1
//                            truncated to (-inf, 0)  if y_i = 0,
//     m_i = mu_i + offset_i.
//
// WHAT THIS TEST VERIFIES, AND WHY THIS TARGET
// --------------------------------------------
// The full conditional above is available in CLOSED FORM, so no weaker
// device (invariance of a kernel, exact enumeration, a detailed-balance
// identity) is needed here: we compare the drawn z_i against the exact
// truncated-normal law itself. The header's own conditional-independence
// argument makes this a complete statement about the block -- given
// (y, mu, offset) the z_i are independent across i and independent across
// successive step() calls, so repeated step() calls produce i.i.d. draws
// from the target and every empirical summary below has an i.i.d. sampling
// distribution with a computable standard error.
//
// For each coordinate we check four things:
//   (1) SUPPORT. Every draw obeys the hard truncation (z > 0 when y = 1,
//       z < 0 when y = 0) and is finite. Analytic bound, zero tolerance.
//   (2) MEAN. E[z] = m + s * lambda(a), where s = +1 if y = 1 and s = -1 if
//       y = 0, a = -s*m is the standardized truncation bound, and
//       lambda(a) = phi(a) / Q(a) is the inverse Mills ratio with
//       Q(a) = P(N(0,1) > a).
//   (3) VARIANCE. Var[z] = 1 + a*lambda(a) - lambda(a)^2.
//   (4) WHOLE DISTRIBUTION. The probability-integral transform
//       u = 1 - Q(w)/Q(a) of the standardized draw w = s*(z - m) must be
//       Uniform(0,1); we test that with a Kolmogorov-Smirnov statistic.
//       Moments (2)-(3) alone cannot detect a sampler that gets the first
//       two moments right and the shape wrong, so the KS check is what
//       actually pins the law down.
//
// Central moments of W ~ TN(0, 1, [a, +inf)) come from the standard
// recursion E[W^k] = (k-1) E[W^(k-2)] + a^(k-1) lambda(a) (integration by
// parts of w^(k-1) * w phi(w)):
//     m1 = lambda,  m2 = 1 + a*lambda,  m3 = (2 + a^2) * lambda,
//     m4 = 3*m2 + a^3*lambda,
//     c2 = m2 - m1^2,   c4 = m4 - 4*m1*m3 + 6*m1^2*m2 - 3*m1^4.
// z is m + s*W, so Var[z] = c2 and the 4th central moment of z is c4 (even
// central moments are invariant under the shift and the sign flip). These
// closed forms were verified against numerical quadrature of the truncated
// density to 10 significant digits at every bound used below.
//
// FIXTURE
// -------
// Ten coordinates over three block instances, chosen to exercise BOTH
// branches of the Robert (1995) truncated-normal sampler inside the block
// (plain N(0,1) rejection for bound a < 0.5, exponential rejection for
// a >= 0.5), including the branch boundary a = 0.5 exactly and a deep tail
// a = 3 where plain rejection would need ~740 proposals per draw:
//
//   Block A (no offset), n = 6:
//     y=1 mu= 0.0 -> a =  0.0   rejection branch
//     y=1 mu= 2.0 -> a = -2.0   rejection branch, nearly untruncated
//     y=0 mu=-1.0 -> a = -1.0   rejection branch
//     y=1 mu=-3.0 -> a =  3.0   exponential branch, deep tail
//     y=0 mu= 1.0 -> a =  1.0   exponential branch
//     y=0 mu= 0.5 -> a =  0.5   exponential branch, exactly at the boundary
//   Block B (scalar offset 1.0 broadcast), n = 2: m = 1.5 and 0.5.
//   Block C (per-observation offset {2.0, -2.0}), n = 2: m = 2.0 and -2.0.
//
// Blocks B and C exist so that the offset is not merely "accepted" but
// shown to move the target: an ignored offset shifts the mean of B[0] by
// 1.0, which is 227 standard errors at this sample size, and a scalar-style
// broadcast of a per-observation offset shifts C[1] by 4.0.
//
// TOLERANCES
// ----------
// Draws are i.i.d., so every threshold is a standard error at N = 40000:
//   mean:  |mean_hat - E[z]|  < 5 * sqrt(c2 / N)
//   var:   |var_hat  - c2|    < 5 * sqrt((c4 - c2^2) / N)   [asymptotic
//          variance of the sample variance for i.i.d. draws]
//   KS:    D_n < 2.2253 / sqrt(N), the level-1e-4 asymptotic critical
//          value (2*exp(-2 x^2) = 1e-4 gives x = 2.2253).
// A 5-sigma moment threshold is a per-check false-alarm rate of 5.7e-7;
// across 10 coordinates the whole test has a false-alarm rate near 1e-3,
// dominated by the KS checks. Nothing here is tuned to the observed
// numbers: the thresholds are the sampling errors implied by N alone.
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
#include "AI4BayesCode/probit_aug_block.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <string>
#include <vector>

using AI4BayesCode::block_context;
using AI4BayesCode::probit_aug_block;
using AI4BayesCode::probit_aug_block_config;

namespace {

const std::size_t N_DRAWS = 40000;

// Upper tail of the standard normal, P(N(0,1) > x). erfc is accurate deep
// into the tail, which matters for the a = 3 coordinate.
inline double norm_upper(double x) {
    return 0.5 * std::erfc(x / std::sqrt(2.0));
}

inline double norm_pdf(double x) {
    return std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
}

// Exact moments of W ~ TN(0, 1, [a, +inf)): mean, 2nd and 4th CENTRAL
// moments, via E[W^k] = (k-1) E[W^(k-2)] + a^(k-1) lambda.
struct tn_moments {
    double mean;   // E[W]
    double c2;     // Var[W]
    double c4;     // E[(W - E W)^4]
};

tn_moments tn_lower_moments(double a) {
    const double lambda = norm_pdf(a) / norm_upper(a);
    const double m1 = lambda;
    const double m2 = 1.0 + a * lambda;
    const double m3 = (2.0 + a * a) * lambda;
    const double m4 = 3.0 * m2 + a * a * a * lambda;
    tn_moments out;
    out.mean = m1;
    out.c2   = m2 - m1 * m1;
    out.c4   = m4 - 4.0 * m1 * m3 + 6.0 * m1 * m1 * m2 - 3.0 * m1 * m1 * m1 * m1;
    return out;
}

// One coordinate of the fixture: y, the effective conditional mean
// m = mu + offset, and the samples drawn for it.
struct coord_result {
    std::string label;
    double y;
    double m;
    double samp_mean, exp_mean, mean_err, mean_tol;
    double samp_var,  exp_var,  var_err,  var_tol;
    double ks_stat,   ks_crit;
    bool   support_ok;
    bool   pass;
};

coord_result check_coord(const std::string& label, double y, double m,
                         std::vector<double>& z) {
    coord_result r;
    r.label = label;
    r.y = y;
    r.m = m;

    const double s = (y >= 0.5) ? 1.0 : -1.0;   // z = m + s * W
    const double a = -s * m;                    // standardized lower bound on W

    // ---- (1) hard support ------------------------------------------------
    r.support_ok = true;
    for (double v : z) {
        if (!std::isfinite(v)) { r.support_ok = false; break; }
        if (y >= 0.5) { if (!(v > 0.0)) { r.support_ok = false; break; } }
        else          { if (!(v < 0.0)) { r.support_ok = false; break; } }
    }

    const double n = static_cast<double>(z.size());

    // ---- sample mean / variance -----------------------------------------
    double mean = 0.0;
    for (double v : z) mean += v;
    mean /= n;
    double var = 0.0;
    for (double v : z) { const double d = v - mean; var += d * d; }
    var /= (n - 1.0);

    // ---- (2)-(3) analytic targets and standard-error tolerances ---------
    const tn_moments tm = tn_lower_moments(a);
    r.samp_mean = mean;
    r.exp_mean  = m + s * tm.mean;
    r.mean_err  = std::abs(mean - r.exp_mean);
    r.mean_tol  = 5.0 * std::sqrt(tm.c2 / n);

    r.samp_var  = var;
    r.exp_var   = tm.c2;
    r.var_err   = std::abs(var - tm.c2);
    r.var_tol   = 5.0 * std::sqrt(std::max(tm.c4 - tm.c2 * tm.c2, 0.0) / n);

    // ---- (4) KS of the probability-integral transform against U(0,1) ----
    // W = s*(z - m) has CDF F(w) = 1 - Q(w)/Q(a) on [a, +inf). Written with
    // upper tails, this stays accurate when Q(a) is tiny (the a = 3 case).
    const double Qa = norm_upper(a);
    std::vector<double> u;
    u.reserve(z.size());
    for (double v : z) {
        const double w = s * (v - m);
        double ui = 1.0 - norm_upper(w) / Qa;
        if (ui < 0.0) ui = 0.0;
        if (ui > 1.0) ui = 1.0;
        u.push_back(ui);
    }
    std::sort(u.begin(), u.end());
    double d_max = 0.0;
    for (std::size_t i = 0; i < u.size(); ++i) {
        const double f_hi = static_cast<double>(i + 1) / n;
        const double f_lo = static_cast<double>(i) / n;
        d_max = std::max(d_max, std::max(f_hi - u[i], u[i] - f_lo));
    }
    r.ks_stat = d_max;
    r.ks_crit = 2.2253 / std::sqrt(n);

    r.pass = r.support_ok &&
             (r.mean_err < r.mean_tol) &&
             (r.var_err  < r.var_tol)  &&
             (r.ks_stat  < r.ks_crit);
    return r;
}

Rcpp::List coord_to_list(const coord_result& r) {
    return Rcpp::List::create(
        Rcpp::Named("label")      = r.label,
        Rcpp::Named("y")          = r.y,
        Rcpp::Named("cond_mean")  = r.m,
        Rcpp::Named("pass")       = r.pass,
        Rcpp::Named("support_ok") = r.support_ok,
        Rcpp::Named("samp_mean")  = r.samp_mean,
        Rcpp::Named("exp_mean")   = r.exp_mean,
        Rcpp::Named("mean_err")   = r.mean_err,
        Rcpp::Named("mean_tol")   = r.mean_tol,
        Rcpp::Named("samp_var")   = r.samp_var,
        Rcpp::Named("exp_var")    = r.exp_var,
        Rcpp::Named("var_err")    = r.var_err,
        Rcpp::Named("var_tol")    = r.var_tol,
        Rcpp::Named("ks_stat")    = r.ks_stat,
        Rcpp::Named("ks_crit")    = r.ks_crit);
}

// Drive one configured block for N_DRAWS steps and return the per-coordinate
// sample paths.
std::vector<std::vector<double> > run_block(probit_aug_block& blk,
                                            std::size_t n_obs,
                                            std::mt19937_64& rng) {
    std::vector<std::vector<double> > out(n_obs);
    for (std::size_t j = 0; j < n_obs; ++j) out[j].reserve(N_DRAWS);
    for (std::size_t it = 0; it < N_DRAWS; ++it) {
        blk.step(rng);
        const arma::vec& v = blk.current();
        for (std::size_t j = 0; j < n_obs; ++j) out[j].push_back(v[j]);
    }
    return out;
}

}  // namespace

// [[Rcpp::export]]
Rcpp::List test_probit_aug_block() {
    std::mt19937_64 rng(20260818);
    std::vector<coord_result> res;

    // ---- Block A: no offset, six regimes --------------------------------
    {
        const arma::vec y  = arma::vec({1.0, 1.0, 0.0, 1.0, 0.0, 0.0});
        const arma::vec mu = arma::vec({0.0, 2.0, -1.0, -3.0, 1.0, 0.5});

        probit_aug_block_config cfg;
        cfg.name   = "z_A";
        cfg.n_obs  = y.n_elem;
        cfg.y_key  = "y";
        cfg.mu_key = "mu";
        probit_aug_block blk(std::move(cfg));

        block_context ctx;
        ctx["y"]  = y;
        ctx["mu"] = mu;
        blk.set_context(ctx);

        std::vector<std::vector<double> > s = run_block(blk, y.n_elem, rng);
        const char* labels[6] = {
            "A0_y1_mu0_bound0",     "A1_y1_mu2_boundm2",
            "A2_y0_mum1_boundm1",   "A3_y1_mum3_bound3",
            "A4_y0_mu1_bound1",     "A5_y0_mu0.5_bound0.5"};
        for (std::size_t j = 0; j < y.n_elem; ++j)
            res.push_back(check_coord(labels[j], y[j], mu[j], s[j]));
    }

    // ---- Block B: scalar offset broadcast -------------------------------
    {
        const arma::vec y   = arma::vec({1.0, 0.0});
        const arma::vec mu  = arma::vec({0.5, -0.5});
        const arma::vec off = arma::vec({1.0});          // length 1 -> broadcast

        probit_aug_block_config cfg;
        cfg.name       = "z_B";
        cfg.n_obs      = y.n_elem;
        cfg.y_key      = "y";
        cfg.mu_key     = "mu";
        cfg.offset_key = "off";
        probit_aug_block blk(std::move(cfg));

        block_context ctx;
        ctx["y"]   = y;
        ctx["mu"]  = mu;
        ctx["off"] = off;
        blk.set_context(ctx);

        std::vector<std::vector<double> > s = run_block(blk, y.n_elem, rng);
        const char* labels[2] = {"B0_scalar_offset_mean1.5",
                                 "B1_scalar_offset_mean0.5"};
        for (std::size_t j = 0; j < y.n_elem; ++j)
            res.push_back(check_coord(labels[j], y[j], mu[j] + off[0], s[j]));
    }

    // ---- Block C: per-observation offset --------------------------------
    {
        const arma::vec y   = arma::vec({1.0, 0.0});
        const arma::vec mu  = arma::vec({0.0, 0.0});
        const arma::vec off = arma::vec({2.0, -2.0});    // length n_obs

        probit_aug_block_config cfg;
        cfg.name       = "z_C";
        cfg.n_obs      = y.n_elem;
        cfg.y_key      = "y";
        cfg.mu_key     = "mu";
        cfg.offset_key = "off";
        probit_aug_block blk(std::move(cfg));

        block_context ctx;
        ctx["y"]   = y;
        ctx["mu"]  = mu;
        ctx["off"] = off;
        blk.set_context(ctx);

        std::vector<std::vector<double> > s = run_block(blk, y.n_elem, rng);
        const char* labels[2] = {"C0_vec_offset_mean2",
                                 "C1_vec_offset_meanm2"};
        for (std::size_t j = 0; j < y.n_elem; ++j)
            res.push_back(check_coord(labels[j], y[j], mu[j] + off[j], s[j]));
    }

    // ---- Aggregate -------------------------------------------------------
    bool all_pass       = true;
    bool all_support_ok = true;
    int  n_fail         = 0;
    Rcpp::List detail(res.size());
    Rcpp::CharacterVector names(res.size());
    for (std::size_t i = 0; i < res.size(); ++i) {
        detail[i] = coord_to_list(res[i]);
        names[i]  = res[i].label;
        if (!res[i].pass)       { all_pass = false; ++n_fail; }
        if (!res[i].support_ok) { all_support_ok = false; }
    }
    detail.attr("names") = names;

    return Rcpp::List::create(
        Rcpp::Named("all_pass")   = all_pass,
        Rcpp::Named("support_ok") = all_support_ok,
        Rcpp::Named("n_coords")   = static_cast<int>(res.size()),
        Rcpp::Named("n_fail")     = n_fail,
        Rcpp::Named("n_draws")    = static_cast<int>(N_DRAWS),
        Rcpp::Named("detail")     = detail);
}

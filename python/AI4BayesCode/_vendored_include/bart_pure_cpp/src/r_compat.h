/*
 *  r_compat.h  —  pure-C++ (R-free) replacements for the small set of
 *  R / Rcpp math + RNG symbols that the vendored BART / genBART / SoftBart
 *  numeric code references.
 *
 *  Copyright (C) 2024-2026 Jungang Zou
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.  A copy of the GPL-2 is available
 *  at https://www.R-project.org/Licenses/GPL-2 .
 *
 *  ---------------------------------------------------------------------
 *  WHY THIS FILE EXISTS
 *
 *  The original BART CRAN code (and its genBART / SoftBart descendants)
 *  was written in a dual-mode style: every file that needs R guards its
 *  R/Rcpp use behind `#ifdef NoRcpp`.  When NoRcpp is defined the code is
 *  meant to compile as standalone C++ — but the actual standalone
 *  implementations of the R math library + R RNG were never shipped in
 *  this tree (the `arn` RNG was hard-wired to R::unif_rand, and the
 *  likelihoods call R::digamma / R::pgamma / Rf_lgammafn directly).
 *
 *  This header fills that gap.  Under `#ifdef NoRcpp` it provides:
 *    * a seedable, thread-local std::mt19937_64 RNG;
 *    * the free RNG functions SoftBart calls bare (unif_rand, norm_rand,
 *      exp_rand);
 *    * a `namespace R { ... }` and the `Rf_*` free functions that the
 *      BART kernel + genBART likelihoods reference, with R-free
 *      implementations (special functions: lgammafn, digamma, trigamma,
 *      pnorm, pgamma, qgamma/qchisq, lbeta, lchoose, dexp, dcauchy;
 *      RNG: rgamma, rnorm, rbeta, rchisq, rgeom, rmultinom, exp_rand).
 *
 *  Effect: the vendored numeric code compiles UNCHANGED with -D NoRcpp.
 *  The R build path (NoRcpp undefined) is completely untouched — this
 *  file is a no-op there, so dual-mode is preserved and the existing
 *  Rcpp wrappers keep working.
 *
 *  Reproducibility: seed the standalone stream once via
 *  bart_rng::set_seed(uint64_t) before sampling.  This replaces R-level
 *  set.seed().  The numerical stream is NOT bit-identical to R's (a
 *  different PRNG + different special-function code), but is
 *  statistically equivalent for BART sampling.
 *  ---------------------------------------------------------------------
 */

#ifndef BART_R_COMPAT_H_
#define BART_R_COMPAT_H_

#ifdef NoRcpp   // entire file is a no-op under the Rcpp (R) build

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>
#include <limits>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884
#endif
#ifndef M_SQRT1_2
#define M_SQRT1_2 0.707106781186547524400844362104849039
#endif
#ifndef M_2PI
#define M_2PI 6.283185307179586476925286766559006
#endif

// =====================================================================
//  RNG core — one seedable thread-local engine
// =====================================================================
namespace bart_rng {

inline std::mt19937_64& engine() {
  // Deterministic default seed; override via set_seed().
  static thread_local std::mt19937_64 eng(20240920ULL);
  return eng;
}

inline void set_seed(std::uint64_t s) { engine().seed(s); }

// uniform on the OPEN interval (0,1) — matches R's unif_rand(), which
// never returns exactly 0 or 1 (important: callers take log(unif_rand)).
inline double runif_open() {
  static thread_local std::uniform_real_distribution<double> u(0.0, 1.0);
  double x;
  do { x = u(engine()); } while (x <= 0.0 || x >= 1.0);
  return x;
}
inline double rnorm_std() {
  std::normal_distribution<double> d(0.0, 1.0);
  return d(engine());
}
inline double rexp_std() {           // mean 1 (rate 1), like R's exp_rand()
  std::exponential_distribution<double> d(1.0);
  return d(engine());
}
inline double rgamma_sc(double shape, double scale) {  // mean = shape*scale
  if (shape <= 0.0) return 0.0;
  std::gamma_distribution<double> d(shape, scale);
  return d(engine());
}

} // namespace bart_rng

// =====================================================================
//  Special functions (deterministic) — R-free implementations
// =====================================================================
namespace bart_sf {

inline double lgammafn(double x) { return std::lgamma(x); }

// digamma psi(x): recurrence up to x>=6 then asymptotic series.
inline double digamma(double x) {
  double result = 0.0;
  if (x <= 0.0 && x == std::floor(x))
    return std::numeric_limits<double>::quiet_NaN();   // poles at 0,-1,-2,...
  // reflection for negative x (rarely needed here)
  if (x < 0.0) {
    return digamma(1.0 - x) - M_PI / std::tan(M_PI * x);
  }
  while (x < 13.0) { result -= 1.0 / x; x += 1.0; }
  const double f = 1.0 / (x * x);
  result += std::log(x) - 0.5 / x
          - f * (1.0/12.0 - f * (1.0/120.0 - f * (1.0/252.0 - f * (1.0/240.0))));
  return result;
}

// trigamma psi'(x): recurrence + asymptotic series.
inline double trigamma(double x) {
  double result = 0.0;
  if (x <= 0.0 && x == std::floor(x))
    return std::numeric_limits<double>::quiet_NaN();
  if (x < 0.0) {
    const double s = M_PI / std::sin(M_PI * x);
    return -trigamma(1.0 - x) + s * s;
  }
  while (x < 13.0) { result += 1.0 / (x * x); x += 1.0; }
  const double f = 1.0 / (x * x);
  result += 1.0 / x + 0.5 * f
          + (f / x) * (1.0/6.0 - f * (1.0/30.0 - f * (1.0/42.0 - f * (1.0/30.0))));
  return result;
}

inline double lbeta(double a, double b) {
  return std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b);
}
inline double lchoose(double n, double k) {
  // R's generalized lchoose; for our integer-ish usage lgamma form is fine.
  return std::lgamma(n + 1.0) - std::lgamma(k + 1.0) - std::lgamma(n - k + 1.0);
}

// --- normal CDF -------------------------------------------------------
inline double pnorm(double x, double mu = 0.0, double sigma = 1.0,
                    int lower_tail = 1, int log_p = 0) {
  const double z = (x - mu) / sigma;
  double p = 0.5 * std::erfc(-z * M_SQRT1_2);     // lower-tail P(Z<=z)
  if (!lower_tail) p = 1.0 - p;
  return log_p ? std::log(p) : p;
}
inline double dnorm(double x, double mu = 0.0, double sigma = 1.0,
                    int give_log = 0) {
  const double z = (x - mu) / sigma;
  const double ld = -0.5 * z * z - std::log(sigma) - 0.5 * std::log(2.0 * M_PI);
  return give_log ? ld : std::exp(ld);
}

// --- regularized incomplete gamma  (Numerical Recipes) ---------------
inline double gser(double a, double x) {        // series, returns P(a,x)
  if (x <= 0.0) return 0.0;
  double ap = a, sum = 1.0 / a, del = sum;
  for (int n = 0; n < 300; ++n) {
    ap += 1.0;
    del *= x / ap;
    sum += del;
    if (std::fabs(del) < std::fabs(sum) * 1e-16) break;
  }
  return sum * std::exp(-x + a * std::log(x) - std::lgamma(a));
}
inline double gcf(double a, double x) {         // continued fraction, returns Q(a,x)
  const double FPMIN = 1e-300;
  double b = x + 1.0 - a, c = 1.0 / FPMIN, d = 1.0 / b, h = d;
  for (int i = 1; i < 300; ++i) {
    const double an = -1.0 * i * (i - a);
    b += 2.0;
    d = an * d + b; if (std::fabs(d) < FPMIN) d = FPMIN;
    c = b + an / c; if (std::fabs(c) < FPMIN) c = FPMIN;
    d = 1.0 / d;
    const double del = d * c;
    h *= del;
    if (std::fabs(del - 1.0) < 1e-16) break;
  }
  return std::exp(-x + a * std::log(x) - std::lgamma(a)) * h;
}
// pgamma with rate=1/scale, R signature: pgamma(q, shape, scale, lower, logp)
inline double pgamma(double q, double shape, double scale = 1.0,
                     int lower_tail = 1, int log_p = 0) {
  const double a = shape, xx = q / scale;
  double P, Q;
  if (xx <= 0.0)       { P = 0.0; Q = 1.0; }
  else if (xx < a + 1) { P = gser(a, xx); Q = 1.0 - P; }
  else                 { Q = gcf(a, xx);  P = 1.0 - Q; }
  const double v = lower_tail ? P : Q;
  return log_p ? std::log(v) : v;
}

// (defined just below; needed by invgammp)
inline double gammp(double a, double x);

// inverse of regularized lower incomplete gamma: returns x with P(a,x)=p
inline double invgammp(double p, double a) {
  if (p <= 0.0) return 0.0;
  if (p >= 1.0) return std::numeric_limits<double>::max();
  const double gln = std::lgamma(a);
  const double a1 = a - 1.0;
  double x, lna1 = 0.0, afac = 0.0;
  if (a > 1.0) {
    lna1 = std::log(a1);
    afac = std::exp(a1 * (lna1 - 1.0) - gln);
    const double pp = (p < 0.5) ? p : 1.0 - p;
    const double t = std::sqrt(-2.0 * std::log(pp));
    double xx = (2.30753 + t * 0.27061) / (1.0 + t * (0.99229 + t * 0.04481)) - t;
    if (p < 0.5) xx = -xx;
    x = std::max(1e-3, a * std::pow(1.0 - 1.0/(9.0*a) - xx/(3.0*std::sqrt(a)), 3.0));
  } else {
    const double t = 1.0 - a * (0.253 + a * 0.12);
    if (p < t) x = std::pow(p / t, 1.0 / a);
    else       x = 1.0 - std::log(1.0 - (p - t) / (1.0 - t));
  }
  for (int j = 0; j < 16; ++j) {
    if (x <= 0.0) return 0.0;
    const double err = gammp(a, x) - p;
    double t2;
    if (a > 1.0) t2 = afac * std::exp(-(x - a1) + a1 * (std::log(x) - lna1));
    else         t2 = std::exp(-x + a1 * std::log(x) - gln);
    const double u = err / t2;
    double dx = u / (1.0 - 0.5 * std::min(1.0, u * ((a - 1.0) / x - 1.0)));
    x -= dx;
    if (x <= 0.0) x = 0.5 * (x + dx);
    if (std::fabs(dx) < 1e-10 * x) break;
  }
  return x;
}
// forward-declare helper used above before its definition order
inline double gammp(double a, double x) {
  if (x <= 0.0 || a <= 0.0) return 0.0;
  return (x < a + 1.0) ? gser(a, x) : 1.0 - gcf(a, x);
}

inline double qgamma(double p, double shape, double scale = 1.0,
                     int lower_tail = 1, int log_p = 0) {
  if (log_p) p = std::exp(p);
  if (!lower_tail) p = 1.0 - p;
  return scale * invgammp(p, shape);
}
inline double qchisq(double p, double df, int lower_tail = 1, int log_p = 0) {
  // chi-square(df) == gamma(shape = df/2, scale = 2)
  return qgamma(p, df / 2.0, 2.0, lower_tail, log_p);
}

inline double dexp_scale(double x, double scale, int give_log) {
  // R's Rf_dexp(x, scale, log): mean = scale, rate = 1/scale
  if (x < 0.0) return give_log ? -std::numeric_limits<double>::infinity() : 0.0;
  const double ld = -std::log(scale) - x / scale;
  return give_log ? ld : std::exp(ld);
}
inline double dcauchy(double x, double loc, double scale, int give_log) {
  const double z = (x - loc) / scale;
  const double dens = 1.0 / (M_PI * scale * (1.0 + z * z));
  return give_log ? std::log(dens) : dens;
}

} // namespace bart_sf

// =====================================================================
//  RNG draws that depend on the special functions (placed after bart_sf)
// =====================================================================
namespace bart_rng {

inline double rchisq(double df) {
  std::chi_squared_distribution<double> d(df);
  return d(engine());
}
inline double rbeta(double a, double b) {
  const double x1 = rgamma_sc(a, 1.0), x2 = rgamma_sc(b, 1.0);
  return x1 / (x1 + x2);
}
inline double rgeom(double p) {
  if (p >= 1.0) return 0.0;
  std::geometric_distribution<long> d(p);   // # failures before 1st success
  return static_cast<double>(d(engine()));
}
// R's rmultinom(n, prob[k], k, rN[k]) via sequential conditional binomials.
inline void rmultinom(int n, const double* prob, int k, int* rN) {
  double norm = 0.0;
  for (int j = 0; j < k; ++j) norm += prob[j];
  int remaining = n;
  double rem_norm = norm;
  for (int j = 0; j < k - 1; ++j) {
    if (remaining <= 0 || rem_norm <= 0.0) { rN[j] = 0; continue; }
    double pj = prob[j] / rem_norm;
    if (pj > 1.0) pj = 1.0;
    if (pj < 0.0) pj = 0.0;
    std::binomial_distribution<int> d(remaining, pj);
    rN[j] = d(engine());
    remaining -= rN[j];
    rem_norm  -= prob[j];
  }
  rN[k - 1] = remaining;
}

} // namespace bart_rng

// =====================================================================
//  Public R-API surface expected by the vendored code
// =====================================================================

// --- bare free functions (SoftBart calls these unqualified) ----------
inline double unif_rand() { return bart_rng::runif_open(); }
inline double norm_rand() { return bart_rng::rnorm_std(); }
inline double exp_rand()  { return bart_rng::rexp_std(); }

// --- global ::pnorm / ::dnorm (BART/lambda.h does `using ::pnorm;`) ---
inline double pnorm(double x, double mu, double sigma, int lower, int logp)
                                                   { return bart_sf::pnorm(x, mu, sigma, lower, logp); }
inline double dnorm(double x, double mu, double sigma, int give_log)
                                                   { return bart_sf::dnorm(x, mu, sigma, give_log); }

// --- SoftBart vendored uses bare Rcout / stop (normally from `using
//     namespace Rcpp`).  Provide R-free equivalents at global scope so the
//     vendored code compiles unchanged under NoRcpp.
inline std::ostream& Rcout = std::cout;                 // C++17 inline variable
[[noreturn]] inline void stop(const std::string& msg) { throw std::runtime_error(msg); }

// --- Rf_* free functions ---------------------------------------------
inline double Rf_lgammafn(double x)               { return bart_sf::lgammafn(x); }
inline double Rf_lbeta(double a, double b)         { return bart_sf::lbeta(a, b); }
inline double Rf_rgamma(double shape, double scale){ return bart_rng::rgamma_sc(shape, scale); }
inline double Rf_rnorm(double mu, double sd)       { return mu + sd * bart_rng::rnorm_std(); }
inline double Rf_rbeta(double a, double b)         { return bart_rng::rbeta(a, b); }
inline double Rf_dexp(double x, double scale, int give_log)
                                                   { return bart_sf::dexp_scale(x, scale, give_log); }
inline double Rf_dcauchy(double x, double loc, double scale, int give_log)
                                                   { return bart_sf::dcauchy(x, loc, scale, give_log); }

// --- namespace R { ... }  (Rcpp-sugar-style qualified calls) ----------
namespace R {
  // RNG
  inline double norm_rand()           { return bart_rng::rnorm_std(); }
  inline double unif_rand()           { return bart_rng::runif_open(); }
  inline double exp_rand()            { return bart_rng::rexp_std(); }
  inline double rchisq(double df)     { return bart_rng::rchisq(df); }
  inline double rgamma(double a, double s) { return bart_rng::rgamma_sc(a, s); }
  inline double rbeta(double a, double b)  { return bart_rng::rbeta(a, b); }
  inline double rgeom(double p)       { return bart_rng::rgeom(p); }
  inline void   rmultinom(int n, double* prob, int k, int* rN)
                                      { bart_rng::rmultinom(n, prob, k, rN); }
  // special functions
  inline double digamma(double x)     { return bart_sf::digamma(x); }
  inline double trigamma(double x)    { return bart_sf::trigamma(x); }
  inline double lgammafn(double x)    { return bart_sf::lgammafn(x); }
  inline double lbeta(double a, double b)  { return bart_sf::lbeta(a, b); }
  inline double lchoose(double n, double k){ return bart_sf::lchoose(n, k); }
  inline double pnorm(double x, double mu = 0.0, double sd = 1.0,
                      int lower = 1, int logp = 0)
                                      { return bart_sf::pnorm(x, mu, sd, lower, logp); }
  inline double dnorm(double x, double mu = 0.0, double sd = 1.0, int logp = 0)
                                      { return bart_sf::dnorm(x, mu, sd, logp); }
  inline double pgamma(double q, double shape, double scale = 1.0,
                       int lower = 1, int logp = 0)
                                      { return bart_sf::pgamma(q, shape, scale, lower, logp); }
  inline double qgamma(double p, double shape, double scale = 1.0,
                       int lower = 1, int logp = 0)
                                      { return bart_sf::qgamma(p, shape, scale, lower, logp); }
  inline double qchisq(double p, double df, int lower = 1, int logp = 0)
                                      { return bart_sf::qchisq(p, df, lower, logp); }
} // namespace R

#endif // NoRcpp
#endif // BART_R_COMPAT_H_

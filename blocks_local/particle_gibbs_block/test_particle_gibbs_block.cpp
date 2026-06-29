/*================================================================================
 *  test_particle_gibbs_block.cpp -- ground-truth library test for
 *  particle_gibbs_block (T0-T4 ladder). Zero external deps beyond armadillo +
 *  the block header. Prints per-regime PASS/FAIL with the tolerance inline;
 *  main() returns non-zero if ANY regime fails.
 *
 *  Copyright (C) 2026 AI4BayesCode contributors. GPL-3.0-or-later.
 *
 *  LADDER (mechanism = gradient-free structured MCMC; cSMC/PGAS path sampler):
 *    T0  sanity         linear-Gaussian, T=1: empirical mean/var of x_1 vs the
 *                       closed-form conjugate Gaussian posterior.
 *    T1a parity         linear-Gaussian, T=20: empirical per-time mean+var vs the
 *                       EXACT RTS (Kalman) smoother. This is the engine-correct
 *                       closed form for a path sampler (no dense inverse faked).
 *    T1b FD-gradient    N/A -- block has NO hand-written gradient (gradient-free
 *                       cSMC). Check #12 / T1b does not fire (stated, not skipped).
 *    T2  recovery       stochastic volatility, T=200, theta fixed at truth:
 *                       90% credible-interval coverage of the true latent path
 *                       ~ 0.90, and posterior mean beats the prior level.
 *    T3  cross-chain    linear-Gaussian, T=50: two DIFFERENT-seed chains from
 *        R-hat<1.01     OVER-DISPERSED inits; cross-chain rank-normalized R-hat
 *                       (Vehtari 2021 rank-norm + classical between-chain, NOT
 *                       split) across all marginals stays < 1.01 (FIXED library
 *                       bar, not user-selectable).
 *    T4  degeneracy     stochastic volatility, T=400, N=8: vanilla PG vs PGAS on
 *        contrast       the earliest state x_0. Vanilla PG path degeneracy ->
 *                       cross-chain R-hat decisively broken (>1.3); ancestor
 *                       sampling (the default) restores mixing (<1.1). Decisive
 *                       contrast, the cSMC analogue of centered-vs-NCR funnel.
 *
 *  SE basis (recorded per validate.md SS5):
 *   - T0/T1a draws are an MCMC chain -> SE uses the EFFECTIVE sample size
 *     (SE(mean)=sqrt(Var/ESS), SE(var)=sqrt(2 Var^2/ESS)), NOT the iid count.
 *   - parity bars use a 6-sigma family-wise slack (T1a has ~40 mean+var checks;
 *     per-entry 6-sigma -> family-wise false-reject ~4e-8). Reports max-z.
 *   - T1b FD tolerance: N/A (no gradient).
 *   - T3/T4 R-hat: rank-normalized cross-chain (NON-split) R-hat; bar 1.01 (T3,
 *     fixed library policy) / decisive >1.3-vs-<1.1 contrast (T4).
 *================================================================================*/

#include "particle_gibbs_block.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using AI4BayesCode::block_context;
using AI4BayesCode::particle_gibbs_block;
using AI4BayesCode::particle_gibbs_block_config;
using AI4BayesCode::pg_resampling_scheme;

// ----------------------------------------------------------------------------
//  Small helpers
// ----------------------------------------------------------------------------

static arma::vec sv1(double x) { arma::vec v(1); v[0] = x; return v; }

static double mean_v(const std::vector<double>& x) {
    double s = 0.0; for (double v : x) s += v; return s / static_cast<double>(x.size());
}
static double var_v(const std::vector<double>& x) {
    const double m = mean_v(x); double s = 0.0;
    for (double v : x) s += (v - m) * (v - m);
    return s / static_cast<double>(x.size() - 1);
}
static double quantile_v(std::vector<double> x, double p) {
    std::sort(x.begin(), x.end());
    const double idx = p * static_cast<double>(x.size() - 1);
    const int lo = static_cast<int>(std::floor(idx));
    const double frac = idx - lo;
    if (lo + 1 < static_cast<int>(x.size())) return x[lo] * (1 - frac) + x[lo + 1] * frac;
    return x[lo];
}

// Autocorrelation-based effective sample size (initial-positive-sequence trunc).
static double ess_v(const std::vector<double>& x) {
    const int n = static_cast<int>(x.size());
    const double m = mean_v(x);
    double v0 = 0.0; for (double v : x) v0 += (v - m) * (v - m); v0 /= n;
    if (v0 <= 0.0) return static_cast<double>(n);
    double sum_rho = 0.0;
    const int maxlag = std::min(n - 1, 1000);
    for (int k = 1; k <= maxlag; ++k) {
        double c = 0.0;
        for (int i = 0; i + k < n; ++i) c += (x[i] - m) * (x[i + k] - m);
        c /= n;
        const double rho = c / v0;
        if (rho < 0.0) break;
        sum_rho += rho;
    }
    double tau = 1.0 + 2.0 * sum_rho;
    if (tau < 1.0) tau = 1.0;
    return static_cast<double>(n) / tau;
}

// Acklam's inverse normal CDF (|err| < 1.2e-9 over the central region used here).
static double inv_norm_cdf(double p) {
    static const double a[] = {-3.969683028665376e+01, 2.209460984245205e+02,
        -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01,
        2.506628277459239e+00};
    static const double b[] = {-5.447609879822406e+01, 1.615858368580409e+02,
        -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01};
    static const double c[] = {-7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00,
        2.938163982698783e+00};
    static const double d[] = {7.784695709041462e-03, 3.224671290700398e-01,
        2.445134137142996e+00, 3.754408661907416e+00};
    const double plow = 0.02425, phigh = 1.0 - plow;
    double q, r;
    if (p < plow) {
        q = std::sqrt(-2.0 * std::log(p));
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    } else if (p <= phigh) {
        q = p - 0.5; r = q * q;
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0);
    } else {
        q = std::sqrt(-2.0 * std::log(1.0 - p));
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0);
    }
}

// Average-rank (1-based) of a pooled vector, ties averaged.
static std::vector<double> rank_avg(const std::vector<double>& v) {
    const int n = static_cast<int>(v.size());
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int i, int j) { return v[i] < v[j]; });
    std::vector<double> r(n);
    int i = 0;
    while (i < n) {
        int j = i;
        while (j + 1 < n && v[idx[j + 1]] == v[idx[i]]) ++j;
        const double avg = ((i + 1) + (j + 1)) / 2.0;
        for (int k = i; k <= j; ++k) r[idx[k]] = avg;
        i = j + 1;
    }
    return r;
}

// Cross-chain rank-normalized R-hat over TWO equal-length chains (NON-split).
static double rank_rhat2(const std::vector<double>& c1, const std::vector<double>& c2) {
    const int M = static_cast<int>(c1.size());
    std::vector<double> pool;
    pool.reserve(2 * M);
    pool.insert(pool.end(), c1.begin(), c1.end());
    pool.insert(pool.end(), c2.begin(), c2.end());
    const std::vector<double> rk = rank_avg(pool);
    const int n2 = 2 * M;
    std::vector<double> z(n2);
    for (int i = 0; i < n2; ++i) z[i] = inv_norm_cdf((rk[i] - 0.5) / n2);
    double m1 = 0.0, m2 = 0.0;
    for (int i = 0; i < M; ++i) m1 += z[i];
    for (int i = M; i < n2; ++i) m2 += z[i];
    m1 /= M; m2 /= M;
    const double gm = 0.5 * (m1 + m2);
    double v1 = 0.0, v2 = 0.0;
    for (int i = 0; i < M; ++i) v1 += (z[i] - m1) * (z[i] - m1);
    for (int i = M; i < n2; ++i) v2 += (z[i] - m2) * (z[i] - m2);
    v1 /= (M - 1); v2 /= (M - 1);
    const double W = 0.5 * (v1 + v2);
    const double B = M * ((m1 - gm) * (m1 - gm) + (m2 - gm) * (m2 - gm));  // C-1 = 1
    if (W <= 0.0) return 1.0;
    const double var_plus = ((M - 1.0) / M) * W + B / M;
    return std::sqrt(var_plus / W);
}

// ----------------------------------------------------------------------------
//  Models: linear-Gaussian (closed-form smoother) and stochastic volatility
// ----------------------------------------------------------------------------
//  Linear-Gaussian SSM, theta read from ctx keys a/q/c/r/m0/P0/y:
//      x_1 ~ N(m0,P0); x_t = a x_{t-1} + N(0,q); y_t = c x_t + N(0,r).

static arma::vec lg_init(const block_context& ctx, std::mt19937_64& rng) {
    const double m0 = ctx.at("m0")[0], P0 = ctx.at("P0")[0];
    std::normal_distribution<double> n01(0.0, 1.0);
    return sv1(m0 + std::sqrt(P0) * n01(rng));
}
static arma::vec lg_trans(int /*t*/, const arma::vec& xp, const block_context& ctx,
                          std::mt19937_64& rng) {
    const double a = ctx.at("a")[0], q = ctx.at("q")[0];
    std::normal_distribution<double> n01(0.0, 1.0);
    return sv1(a * xp[0] + std::sqrt(q) * n01(rng));
}
static double lg_obs(int t, const arma::vec& x, const block_context& ctx) {
    const double c = ctx.at("c")[0], r = ctx.at("r")[0], y = ctx.at("y")[t];
    const double dlt = y - c * x[0];
    return -0.5 * std::log(2.0 * M_PI * r) - dlt * dlt / (2.0 * r);
}
static double lg_tlpdf(int /*t*/, const arma::vec& x, const arma::vec& xp,
                       const block_context& ctx) {
    const double a = ctx.at("a")[0], q = ctx.at("q")[0];
    const double dlt = x[0] - a * xp[0];
    return -0.5 * std::log(2.0 * M_PI * q) - dlt * dlt / (2.0 * q);
}

//  Stochastic volatility, theta read from ctx keys mu/phi/sigma_eta/y:
//      h_1 ~ N(mu, se^2/(1-phi^2)); h_t = mu + phi(h_{t-1}-mu) + N(0,se^2);
//      y_t ~ N(0, exp(h_t)).
static arma::vec sv_init(const block_context& ctx, std::mt19937_64& rng) {
    const double mu = ctx.at("mu")[0], phi = ctx.at("phi")[0], se = ctx.at("sigma_eta")[0];
    const double sd = se / std::sqrt(1.0 - phi * phi);
    std::normal_distribution<double> n01(0.0, 1.0);
    return sv1(mu + sd * n01(rng));
}
static arma::vec sv_trans(int /*t*/, const arma::vec& xp, const block_context& ctx,
                          std::mt19937_64& rng) {
    const double mu = ctx.at("mu")[0], phi = ctx.at("phi")[0], se = ctx.at("sigma_eta")[0];
    std::normal_distribution<double> n01(0.0, 1.0);
    return sv1(mu + phi * (xp[0] - mu) + se * n01(rng));
}
static double sv_obs(int t, const arma::vec& x, const block_context& ctx) {
    const double y = ctx.at("y")[t]; const double h = x[0];
    return -0.5 * (std::log(2.0 * M_PI) + h + y * y * std::exp(-h));
}
static double sv_tlpdf(int /*t*/, const arma::vec& x, const arma::vec& xp,
                       const block_context& ctx) {
    const double mu = ctx.at("mu")[0], phi = ctx.at("phi")[0], se = ctx.at("sigma_eta")[0];
    const double q = se * se;
    const double dlt = x[0] - (mu + phi * (xp[0] - mu));
    return -0.5 * std::log(2.0 * M_PI * q) - dlt * dlt / (2.0 * q);
}

static void simulate_lg(double a, double q, double c, double r, double m0, double P0,
                        int T, std::mt19937_64& rng, arma::vec& x, arma::vec& y) {
    x.set_size(T); y.set_size(T);
    std::normal_distribution<double> n01(0.0, 1.0);
    x[0] = m0 + std::sqrt(P0) * n01(rng);
    for (int t = 1; t < T; ++t) x[t] = a * x[t - 1] + std::sqrt(q) * n01(rng);
    for (int t = 0; t < T; ++t) y[t] = c * x[t] + std::sqrt(r) * n01(rng);
}
static void simulate_sv(double mu, double phi, double se, int T,
                        std::mt19937_64& rng, arma::vec& h, arma::vec& y) {
    h.set_size(T); y.set_size(T);
    std::normal_distribution<double> n01(0.0, 1.0);
    h[0] = mu + (se / std::sqrt(1.0 - phi * phi)) * n01(rng);
    for (int t = 1; t < T; ++t) h[t] = mu + phi * (h[t - 1] - mu) + se * n01(rng);
    for (int t = 0; t < T; ++t) y[t] = std::exp(0.5 * h[t]) * n01(rng);
}

// Exact RTS (Kalman) smoother for the linear-Gaussian SSM.
static void rts_smoother(double a, double q, double c, double r, double m0, double P0,
                         const arma::vec& y, arma::vec& ms, arma::vec& Ps) {
    const int T = static_cast<int>(y.n_elem);
    arma::vec mp(T), Pp(T), mf(T), Pf(T);
    mp[0] = m0; Pp[0] = P0;
    for (int t = 0; t < T; ++t) {
        if (t > 0) { mp[t] = a * mf[t - 1]; Pp[t] = a * a * Pf[t - 1] + q; }
        const double S = c * c * Pp[t] + r;
        const double K = c * Pp[t] / S;
        mf[t] = mp[t] + K * (y[t] - c * mp[t]);
        Pf[t] = (1.0 - K * c) * Pp[t];
    }
    ms.set_size(T); Ps.set_size(T);
    ms[T - 1] = mf[T - 1]; Ps[T - 1] = Pf[T - 1];
    for (int t = T - 2; t >= 0; --t) {
        const double Ppn = a * a * Pf[t] + q;   // P_pred_{t+1}
        const double mpn = a * mf[t];           // m_pred_{t+1}
        const double C = Pf[t] * a / Ppn;
        ms[t] = mf[t] + C * (ms[t + 1] - mpn);
        Ps[t] = Pf[t] + C * C * (Ps[t + 1] - Ppn);
    }
}

// ----------------------------------------------------------------------------
//  Driver: run a chain, collect per-time draws (state_dim == 1).
// ----------------------------------------------------------------------------
static void run_chain(particle_gibbs_block& blk, const block_context& ctx, int T,
                      int burn, int M, std::mt19937_64& rng,
                      std::vector<std::vector<double>>& draws) {
    blk.set_context(ctx);
    for (int s = 0; s < burn; ++s) blk.step(rng);
    draws.assign(T, std::vector<double>());
    for (int t = 0; t < T; ++t) draws[t].reserve(M);
    for (int s = 0; s < M; ++s) {
        blk.step(rng);
        const arma::vec& p = blk.current();
        for (int t = 0; t < T; ++t) draws[t].push_back(p[t]);
    }
}

static particle_gibbs_block_config lg_cfg(const std::string& nm, int T, int N,
                                          bool ancestor) {
    particle_gibbs_block_config cfg;
    cfg.name = nm; cfg.T = T; cfg.state_dim = 1; cfg.n_particles = N;
    cfg.ancestor_sampling = ancestor;
    cfg.resampling = pg_resampling_scheme::SYSTEMATIC;
    cfg.init_sample = lg_init; cfg.transition_sample = lg_trans;
    cfg.obs_loglik = lg_obs; cfg.transition_logpdf = lg_tlpdf;
    return cfg;
}
static particle_gibbs_block_config sv_cfg(const std::string& nm, int T, int N,
                                          bool ancestor) {
    particle_gibbs_block_config cfg;
    cfg.name = nm; cfg.T = T; cfg.state_dim = 1; cfg.n_particles = N;
    cfg.ancestor_sampling = ancestor;
    cfg.resampling = pg_resampling_scheme::SYSTEMATIC;
    cfg.init_sample = sv_init; cfg.transition_sample = sv_trans;
    cfg.obs_loglik = sv_obs; cfg.transition_logpdf = sv_tlpdf;
    return cfg;
}

// ----------------------------------------------------------------------------
//  Regimes
// ----------------------------------------------------------------------------

static bool test_T0() {
    const double m0 = 0.0, P0 = 1.0, c = 1.0, r = 0.5, y1 = 1.0;
    const double post_var = 1.0 / (1.0 / P0 + c * c / r);            // 1/3
    const double post_mean = post_var * (m0 / P0 + c * y1 / r);     // 2/3
    block_context ctx;
    ctx["m0"] = sv1(m0); ctx["P0"] = sv1(P0); ctx["a"] = sv1(0.0);
    ctx["q"] = sv1(1.0); ctx["c"] = sv1(c); ctx["r"] = sv1(r); ctx["y"] = sv1(y1);

    particle_gibbs_block_config cfg = lg_cfg("x", 1, 32, /*ancestor=*/false);
    particle_gibbs_block blk(cfg);
    std::mt19937_64 rng(12345ULL);
    std::vector<std::vector<double>> draws;
    run_chain(blk, ctx, 1, 1000, 20000, rng, draws);

    const double em = mean_v(draws[0]), ev = var_v(draws[0]);
    const double e = ess_v(draws[0]);
    const double se_m = std::sqrt(post_var / e);
    const double se_v = std::sqrt(2.0 * post_var * post_var / e);
    const double zm = std::abs(em - post_mean) / se_m;
    const double zv = std::abs(ev - post_var) / se_v;
    const bool pass = (zm < 5.0) && (zv < 5.0);
    std::printf("  T0 sanity (LG T=1 conjugate): mean %.4f vs %.4f (z=%.2f), "
                "var %.4f vs %.4f (z=%.2f), ESS=%.0f  [bar z<5]  %s\n",
                em, post_mean, zm, ev, post_var, zv, e, pass ? "PASS" : "FAIL");
    return pass;
}

static bool test_T1a() {
    const double a = 0.9, q = 0.3, c = 1.0, r = 0.5, m0 = 0.0, P0 = 1.0;
    const int T = 20;
    std::mt19937_64 sim(20260624ULL);
    arma::vec xtrue, y; simulate_lg(a, q, c, r, m0, P0, T, sim, xtrue, y);
    arma::vec ms, Ps; rts_smoother(a, q, c, r, m0, P0, y, ms, Ps);

    block_context ctx;
    ctx["a"] = sv1(a); ctx["q"] = sv1(q); ctx["c"] = sv1(c); ctx["r"] = sv1(r);
    ctx["m0"] = sv1(m0); ctx["P0"] = sv1(P0); ctx["y"] = y;

    particle_gibbs_block_config cfg = lg_cfg("x", T, 64, /*ancestor=*/true);
    particle_gibbs_block blk(cfg);
    std::mt19937_64 rng(777ULL);
    std::vector<std::vector<double>> draws;
    run_chain(blk, ctx, T, 500, 4000, rng, draws);

    double max_zm = 0.0, max_zv = 0.0;
    for (int t = 0; t < T; ++t) {
        const double em = mean_v(draws[t]), ev = var_v(draws[t]);
        const double e = ess_v(draws[t]);
        const double se_m = std::sqrt(Ps[t] / e);
        const double se_v = std::sqrt(2.0 * Ps[t] * Ps[t] / e);
        max_zm = std::max(max_zm, std::abs(em - ms[t]) / se_m);
        max_zv = std::max(max_zv, std::abs(ev - Ps[t]) / se_v);
    }
    const bool pass = (max_zm < 6.0) && (max_zv < 6.0);
    std::printf("  T1a parity (LG T=20 vs RTS smoother): max mean-z=%.2f, "
                "max var-z=%.2f  [family-wise bar z<6]  %s\n",
                max_zm, max_zv, pass ? "PASS" : "FAIL");
    return pass;
}

static bool test_T2() {
    const double mu = -0.5, phi = 0.95, se = 0.25;
    const int T = 200;
    std::mt19937_64 sim(424242ULL);
    arma::vec htrue, y; simulate_sv(mu, phi, se, T, sim, htrue, y);

    block_context ctx;
    ctx["mu"] = sv1(mu); ctx["phi"] = sv1(phi); ctx["sigma_eta"] = sv1(se); ctx["y"] = y;

    particle_gibbs_block_config cfg = sv_cfg("h", T, 64, /*ancestor=*/true);
    particle_gibbs_block blk(cfg);
    std::mt19937_64 rng(2024ULL);
    std::vector<std::vector<double>> draws;
    run_chain(blk, ctx, T, 500, 3000, rng, draws);

    int covered = 0;
    double sse_post = 0.0, sse_prior = 0.0;
    for (int t = 0; t < T; ++t) {
        const double lo = quantile_v(draws[t], 0.05);
        const double hi = quantile_v(draws[t], 0.95);
        if (htrue[t] >= lo && htrue[t] <= hi) ++covered;
        const double pm = mean_v(draws[t]);
        sse_post += (pm - htrue[t]) * (pm - htrue[t]);
        sse_prior += (mu - htrue[t]) * (mu - htrue[t]);
    }
    const double cov = static_cast<double>(covered) / T;
    const double rmse_post = std::sqrt(sse_post / T);
    const double rmse_prior = std::sqrt(sse_prior / T);
    const bool cov_ok = (cov >= 0.80 && cov <= 0.975);
    const bool info_ok = (rmse_post < 0.95 * rmse_prior);
    const bool pass = cov_ok && info_ok;
    std::printf("  T2 recovery (SV T=200, theta fixed): 90%% coverage=%.3f "
                "[bar 0.80-0.975], RMSE post=%.3f vs prior=%.3f [post<0.95*prior]  %s\n",
                cov, rmse_post, rmse_prior, pass ? "PASS" : "FAIL");
    return pass;
}

static bool test_T3() {
    const double a = 0.9, q = 0.3, c = 1.0, r = 0.5, m0 = 0.0, P0 = 1.0;
    const int T = 50;
    std::mt19937_64 sim(31337ULL);
    arma::vec xtrue, y; simulate_lg(a, q, c, r, m0, P0, T, sim, xtrue, y);

    block_context ctx;
    ctx["a"] = sv1(a); ctx["q"] = sv1(q); ctx["c"] = sv1(c); ctx["r"] = sv1(r);
    ctx["m0"] = sv1(m0); ctx["P0"] = sv1(P0); ctx["y"] = y;

    const int M = 3000, burn = 500;
    particle_gibbs_block_config cfg1 = lg_cfg("x", T, 64, true);
    cfg1.initial_path = arma::vec(T, arma::fill::value(5.0));   // over-dispersed +5
    particle_gibbs_block_config cfg2 = lg_cfg("x", T, 64, true);
    cfg2.initial_path = arma::vec(T, arma::fill::value(-5.0));  // over-dispersed -5

    particle_gibbs_block b1(cfg1), b2(cfg2);
    std::mt19937_64 r1(11ULL), r2(99ULL);
    std::vector<std::vector<double>> d1, d2;
    run_chain(b1, ctx, T, burn, M, r1, d1);
    run_chain(b2, ctx, T, burn, M, r2, d2);

    double max_rhat = 0.0; int arg = -1;
    for (int t = 0; t < T; ++t) {
        const double rh = rank_rhat2(d1[t], d2[t]);
        if (rh > max_rhat) { max_rhat = rh; arg = t; }
    }
    const bool pass = (max_rhat < 1.01);
    std::printf("  T3 cross-chain R-hat (LG T=50, inits +-5, seeds 11/99): "
                "max rank-Rhat=%.4f at t=%d  [bar <1.01]  %s\n",
                max_rhat, arg, pass ? "PASS" : "FAIL");
    return pass;
}

static bool test_T4() {
    const double mu = -0.5, phi = 0.95, se = 0.25;
    const int T = 400, N = 8, M = 2000, burn = 500;
    std::mt19937_64 sim(987654ULL);
    arma::vec htrue, y; simulate_sv(mu, phi, se, T, sim, htrue, y);

    block_context ctx;
    ctx["mu"] = sv1(mu); ctx["phi"] = sv1(phi); ctx["sigma_eta"] = sv1(se); ctx["y"] = y;

    auto rhat_x0 = [&](bool ancestor) {
        particle_gibbs_block_config c1 = sv_cfg("h", T, N, ancestor);
        c1.initial_path = arma::vec(T, arma::fill::value(mu + 3.0));
        particle_gibbs_block_config c2 = sv_cfg("h", T, N, ancestor);
        c2.initial_path = arma::vec(T, arma::fill::value(mu - 3.0));
        particle_gibbs_block b1(c1), b2(c2);
        std::mt19937_64 r1(7ULL), r2(8191ULL);
        std::vector<std::vector<double>> d1, d2;
        run_chain(b1, ctx, T, burn, M, r1, d1);
        run_chain(b2, ctx, T, burn, M, r2, d2);
        return rank_rhat2(d1[0], d2[0]);   // earliest state, most degenerate
    };

    const double rhat_vanilla = rhat_x0(false);
    const double rhat_pgas    = rhat_x0(true);
    const bool pass = (rhat_vanilla > 1.3) && (rhat_pgas < 1.1);
    std::printf("  T4 degeneracy (SV T=400, N=8, x_0 cross-chain R-hat): "
                "vanilla PG=%.3f [must >1.3], PGAS=%.3f [must <1.1]  %s\n",
                rhat_vanilla, rhat_pgas, pass ? "PASS" : "FAIL");
    return pass;
}

// Block-local runnable checks (BL#) folded into the ladder.
static bool test_BL() {
    bool ok = true;
    // BL4 -- conditional SMC needs N>=2; N=1 must throw at construction.
    {
        bool threw = false;
        try {
            particle_gibbs_block_config cfg = lg_cfg("x", 5, 1, false);
            particle_gibbs_block blk(cfg);
            (void) blk;
        } catch (const std::invalid_argument&) { threw = true; }
        ok = ok && threw;
        std::printf("  BL4 (N>=2 guard): N=1 %s  %s\n",
                    threw ? "throws" : "ACCEPTED(!)", threw ? "PASS" : "FAIL");
    }
    // BL5 -- PGAS requires transition_logpdf; missing must throw.
    {
        bool threw = false;
        try {
            particle_gibbs_block_config cfg = lg_cfg("x", 5, 16, true);
            cfg.transition_logpdf = nullptr;  // PGAS on but no logpdf
            particle_gibbs_block blk(cfg);
            (void) blk;
        } catch (const std::invalid_argument&) { threw = true; }
        ok = ok && threw;
        std::printf("  BL5 (transition_logpdf required when ancestor on): %s  %s\n",
                    threw ? "throws" : "ACCEPTED(!)", threw ? "PASS" : "FAIL");
    }
    // BL6 -- first sweep without a reference must NOT return the zero init path.
    {
        const double a = 0.9, q = 0.3, c = 1.0, r = 0.5, m0 = 0.0, P0 = 1.0;
        const int T = 10;
        std::mt19937_64 sim(5ULL);
        arma::vec xtrue, y; simulate_lg(a, q, c, r, m0, P0, T, sim, xtrue, y);
        block_context ctx;
        ctx["a"] = sv1(a); ctx["q"] = sv1(q); ctx["c"] = sv1(c); ctx["r"] = sv1(r);
        ctx["m0"] = sv1(m0); ctx["P0"] = sv1(P0); ctx["y"] = y;
        particle_gibbs_block_config cfg = lg_cfg("x", T, 32, true);  // no initial_path
        particle_gibbs_block blk(cfg);
        blk.set_context(ctx);
        std::mt19937_64 rng(1ULL);
        blk.step(rng);
        const double nrm = arma::norm(blk.current());
        const bool nonzero = (nrm > 1e-9);
        ok = ok && nonzero;
        std::printf("  BL6 (first sweep seeds via unconditional PF, not zeros): "
                    "||path||=%.3f  %s\n", nrm, nonzero ? "PASS" : "FAIL");
    }
    return ok;
}

int main() {
    std::printf("=== particle_gibbs_block library test (T0-T4) ===\n");
    std::printf("(T1b FD-gradient: N/A -- gradient-free cSMC, Check #12 does not fire)\n");
    bool ok = true;
    ok &= test_T0();
    ok &= test_T1a();
    ok &= test_T2();
    ok &= test_T3();
    ok &= test_T4();
    ok &= test_BL();
    std::printf("=== %s ===\n", ok ? "ALL REGIMES PASS" : "SOME REGIME FAILED");
    return ok ? 0 : 1;
}

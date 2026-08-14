// ============================================================================
// Library-standard validation ladder (block_design validate.md T0-T4) for the
// genBART M-step slice sampler (GENBART/slice.h) and its integration into
// draw_tree() / genbart_model.
//
// Context: Linero (2022) Algorithm 2 line 6 prescribes refreshing each leaf
// value from its EXACT full conditional via slice sampling (Neal, 2003). The
// former M-step drew from a Laplace approximation without a Metropolis
// correction (targets the conditional only approximately). This ladder
// validates the replacement to the same standard a NEW library block must
// pass:
//
//   T0  sanity           closed-form Gaussian-conjugate conditional +
//                        empty-leaf (prior) recovery, mean/sd/KS within MC SE
//   T1a parity           engine-correct references: grid-integrated exact
//                        conditionals (Poisson, logistic; Laplace gap printed)
//                        + WHOLE-MODEL pure-prior recovery under a flat
//                        likelihood (the honest surrogate for an intractable
//                        posterior: with log_f == 0 the stationary law of
//                        r(x) is N(0, T*sigma_mu^2) at every x)
//   T1b FD-gradient      SILENT by mechanism: the slice M-step uses only
//                        log_f -- no hand-written gradient exists in the
//                        changed path (stated per validate.md Sec.2)
//   T2  recovery         known-truth Poisson DGP; corr / RMSE gates
//   T3  cross-chain      TWO chains, DIFFERENT seeds, deliberately
//       rank R-hat       over-dispersed inits (chain B warm-started on
//                        SHUFFLED y, then set_Y to the real y); Vehtari-2021
//                        rank-normalized CROSS-chain (non-split) R-hat over
//                        r(x) marginals must be < 1.01
//   T4  stress           block-specific hard regimes: huge leaf (n=5000,
//                        conditional sd << w), wide prior (sigma_mu=5, w >>
//                        conditional sd -> shrinkage stress), all-zero-count
//                        skewed leaf, extreme-lambda finiteness; plus the
//                        vendored-kernel stateful regime: same-seed
//                        determinism and two-instance state isolation
//
// Semantic-check silences stated per validate.md ("do not let silence read
// as a pass"): Check #12/T1b silent (no hand-written gradient); Check #18
// silent (no mass metric); Check #20 N/A (no n_warmup_per_step); Check #14
// N/A (the RJ structure-move ratios are untouched by this change and carry
// their Laplace G-terms per Proposition 1).
//
// All regimes are seeded and deterministic (bart_rng::set_seed).
// ============================================================================
#include <armadillo>
#include "AI4BayesCode/genbart_block.hpp"   // pulls in the genBART kernel
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using genbart::slice_leaf;
using genbart::laplace_leaf;
using genbart::laplace_opts;
using genbart::laplace_proposal;
using genbart::rjmcmc_hypers;
using genbart::genbart_model;

static int fails = 0, checks = 0;
static void ok(bool cond, const char* what, double got, double tol) {
    ++checks; if (!cond) ++fails;
    std::printf("  [%s] %-62s |val|=%.4g (tol %.4g)\n",
                cond ? "PASS" : "FAIL", what, got, tol);
}

// ---------------------------------------------------------------------------
// Helpers: slice chain on one leaf, moments, grid reference, rank R-hat
// ---------------------------------------------------------------------------
static std::vector<double> run_chain(
    const std::vector<double>& ys, const std::vector<double>& lams,
    const std::vector<std::size_t>& idx, double sigma_mu,
    const genbart::likelihood& lik, genbart::rn& gen,
    std::size_t M, std::size_t thin, double mu_start)
{
    std::vector<double> out;
    out.reserve(M / thin);
    double mu = mu_start;
    for (std::size_t m = 0; m < M; ++m) {
        mu = slice_leaf(ys.data(), lams.data(), idx.data(), ys.size(),
                        sigma_mu, lik, mu, gen);
        if ((m + 1) % thin == 0) out.push_back(mu);
    }
    return out;
}
static double mean_of(const std::vector<double>& v) {
    double s = 0; for (double x : v) s += x; return s / v.size();
}
static double sd_of(const std::vector<double>& v, double m) {
    double s = 0; for (double x : v) s += (x - m) * (x - m);
    return std::sqrt(s / (v.size() - 1));
}

// Grid-integrated reference for a 1-D leaf conditional: returns mean, sd and
// a quantile function over [lo, hi] with step h.
struct grid_ref {
    double mean, sd;
    std::vector<double> cdf;
    double lo, h;
    double q(double p) const {
        const std::size_t g = (std::size_t)(std::lower_bound(cdf.begin(), cdf.end(), p) - cdf.begin());
        return lo + std::min(g, cdf.size() - 1) * h;
    }
};
static grid_ref grid_conditional(
    const std::vector<double>& ys, const std::vector<double>& lams,
    const std::vector<std::size_t>& idx, double sigma_mu,
    const genbart::likelihood& lik, double lo, double hi, double h)
{
    const std::size_t n = ys.size();
    const std::size_t G = (std::size_t)((hi - lo) / h) + 1;
    std::vector<double> gv(G);
    double lmax = -1e300;
    for (std::size_t g = 0; g < G; ++g) {
        const double mu = lo + g * h;
        double lp = -0.5 * mu * mu / (sigma_mu * sigma_mu);
        for (std::size_t j = 0; j < n && std::isfinite(lp); ++j)
            lp += lik.log_f(ys[j], lams[j] + mu, idx[j]);
        gv[g] = std::isfinite(lp) ? lp : -1e300;
        lmax = std::max(lmax, gv[g]);
    }
    grid_ref r; r.lo = lo; r.h = h; r.cdf.resize(G);
    double Z = 0, m1 = 0, m2 = 0;
    for (std::size_t g = 0; g < G; ++g) {
        const double f = std::exp(gv[g] - lmax);
        const double x = lo + g * h;
        Z += f; m1 += x * f; m2 += x * x * f;
        r.cdf[g] = Z;
    }
    r.mean = m1 / Z;
    r.sd   = std::sqrt(m2 / Z - r.mean * r.mean);
    for (std::size_t g = 0; g < G; ++g) r.cdf[g] /= Z;
    return r;
}

// Acklam inverse-normal CDF (same approximation as tests/test_simplex_dirichlet.cpp).
static double norm_inv_cdf(double p) {
    static const double a[] = {
        -3.969683028665376e+01,  2.209460984245205e+02,
        -2.759285104469687e+02,  1.383577518672690e+02,
        -3.066479806614716e+01,  2.506628277459239e+00};
    static const double b[] = {
        -5.447609879822406e+01,  1.615858368580409e+02,
        -1.556989798598866e+02,  6.680131188771972e+01,
        -1.328068155288572e+01};
    static const double c[] = {
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00,  2.938163982698783e+00};
    static const double d[] = {
         7.784695709041462e-03,  3.224671290700398e-01,
         2.445134137142996e+00,  3.754408661907416e+00};
    const double p_low = 0.02425, p_high = 1.0 - p_low;
    if (p < p_low) {
        const double q = std::sqrt(-2.0 * std::log(p));
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) /
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0);
    } else if (p <= p_high) {
        const double q = p - 0.5, r = q * q;
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q
             / (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0);
    } else {
        const double q = std::sqrt(-2.0 * std::log(1.0 - p));
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) /
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0);
    }
}

// Vehtari et al. 2021 rank-normalized CROSS-chain (non-split) R-hat:
// rank-normalize the POOLED draws of the two full chains, then take the
// classical between-chain R-hat over the two chains (validate.md T3 form).
static double cross_chain_rank_rhat(const std::vector<double>& c1,
                                    const std::vector<double>& c2)
{
    const std::size_t N = std::min(c1.size(), c2.size());
    const std::size_t S = 2 * N;
    std::vector<std::pair<double, std::size_t> > pooled(S);
    for (std::size_t i = 0; i < N; ++i) { pooled[i] = {c1[i], i}; pooled[N + i] = {c2[i], N + i}; }
    std::sort(pooled.begin(), pooled.end());
    std::vector<double> z(S);
    for (std::size_t r = 0; r < S; ) {
        std::size_t r2 = r;
        while (r2 + 1 < S && pooled[r2 + 1].first == pooled[r].first) ++r2;
        const double avg_rank = 0.5 * (double)(r + r2) + 1.0;   // 1-based average rank
        const double p = (avg_rank - 0.375) / ((double)S + 0.25);
        const double zz = norm_inv_cdf(p);
        for (std::size_t k = r; k <= r2; ++k) z[pooled[k].second] = zz;
        r = r2 + 1;
    }
    double m1 = 0, m2 = 0;
    for (std::size_t i = 0; i < N; ++i) { m1 += z[i]; m2 += z[N + i]; }
    m1 /= N; m2 /= N;
    double w1 = 0, w2 = 0;
    for (std::size_t i = 0; i < N; ++i) {
        w1 += (z[i] - m1) * (z[i] - m1);
        w2 += (z[N + i] - m2) * (z[N + i] - m2);
    }
    w1 /= (N - 1); w2 /= (N - 1);
    const double W  = 0.5 * (w1 + w2);
    const double mb = 0.5 * (m1 + m2);
    const double B  = (double)N * ((m1 - mb) * (m1 - mb) + (m2 - mb) * (m2 - mb)); // M-1 = 1
    const double vh = ((double)N - 1.0) / (double)N * W + B / (double)N;
    return std::sqrt(vh / W);
}

// Flat likelihood: log_f == 0 everywhere. Under it the model's stationary
// law is the BART prior, so r(x) ~ N(0, T * sigma_mu^2) at every x.
struct flat_lik : genbart::likelihood {
    double log_f(double, double, std::size_t) const override { return 0.0; }
    double score(double, double, std::size_t) const override { return 0.0; }
    double obs_info(double, double, std::size_t) const override { return 0.0; }
    const char* name() const override { return "flat"; }
};

int main() {
    genbart::arn gen;

    // =======================================================================
    // T0 -- sanity: closed-form conditionals
    // =======================================================================
    std::printf("================ T0: sanity (closed-form) ================\n");
    bart_rng::set_seed(20260814u);
    {
        const double sigma = 0.7, sigma_mu = 0.3;
        genbart::lik::normal_lik lik(sigma);
        const std::size_t n = 25;
        std::vector<double> ys(n), lams(n);
        std::vector<std::size_t> idx(n);
        for (std::size_t i = 0; i < n; ++i) {
            ys[i]   = 0.4 + 0.05 * (double)i - 0.3 * ((i % 3) == 0);
            lams[i] = 0.1 * std::sin((double)i);
            idx[i]  = i;
        }
        double sr = 0; for (std::size_t i = 0; i < n; ++i) sr += ys[i] - lams[i];
        const double prec = 1.0 / (sigma_mu * sigma_mu) + (double)n / (sigma * sigma);
        const double m_star  = (sr / (sigma * sigma)) / prec;
        const double sd_star = 1.0 / std::sqrt(prec);

        auto d = run_chain(ys, lams, idx, sigma_mu, lik, gen, 200000, 10, 1.5);
        const double mh = mean_of(d), sh = sd_of(d, mh);
        ok(std::abs(mh - m_star) < 0.02 * sd_star,
           "T0 gaussian: |mean - m*| < 0.02 sd*", std::abs(mh - m_star), 0.02 * sd_star);
        ok(std::abs(sh / sd_star - 1.0) < 0.03,
           "T0 gaussian: |sd/sd* - 1| < 0.03", std::abs(sh / sd_star - 1.0), 0.03);
        std::sort(d.begin(), d.end());
        double ks = 0;
        for (std::size_t i = 0; i < d.size(); ++i) {
            const double F = 0.5 * std::erfc(-(d[i] - m_star) / (sd_star * std::sqrt(2.0)));
            ks = std::max(ks, std::max(std::abs(F - (double)(i + 1) / d.size()),
                                       std::abs(F - (double)i / d.size())));
        }
        ok(ks < 0.02, "T0 gaussian: KS vs exact Normal < 0.02", ks, 0.02);
    }
    {   // empty leaf: conditional == prior
        const double sigma_mu = 0.4;
        genbart::lik::normal_lik lik(1.0);
        std::vector<double> ys, lams; std::vector<std::size_t> idx;
        auto d = run_chain(ys, lams, idx, sigma_mu, lik, gen, 200000, 10, 2.0);
        const double mh = mean_of(d), sh = sd_of(d, mh);
        ok(std::abs(mh) < 0.03 * sigma_mu, "T0 empty leaf: |mean| < 0.03 s_mu",
           std::abs(mh), 0.03 * sigma_mu);
        ok(std::abs(sh / sigma_mu - 1.0) < 0.03, "T0 empty leaf: |sd/s_mu - 1| < 0.03",
           std::abs(sh / sigma_mu - 1.0), 0.03);
    }

    // =======================================================================
    // T1a -- parity: grid-integrated exact conditionals + whole-model
    //        pure-prior recovery (flat likelihood)
    // =======================================================================
    std::printf("================ T1a: parity (grid + pure-prior) ================\n");
    {   // Poisson leaf (skewed); Laplace gap printed = the OLD M-step's bias
        const double sigma_mu = 0.5;
        genbart::lik::poisson_lik lik;
        std::vector<double> ys   = {0, 1, 0, 2, 5, 0, 1, 3};
        std::vector<double> lams = {0.2, -0.1, 0.0, 0.3, 0.5, -0.2, 0.1, 0.4};
        std::vector<std::size_t> idx = {0, 1, 2, 3, 4, 5, 6, 7};
        grid_ref ref = grid_conditional(ys, lams, idx, sigma_mu, lik, -4.0, 4.0, 1e-4);
        laplace_opts lo_opts; lo_opts.init_mu = 0.0;
        laplace_proposal q = laplace_leaf(ys.data(), lams.data(), idx.data(),
                                          ys.size(), sigma_mu, lik, lo_opts);
        auto d = run_chain(ys, lams, idx, sigma_mu, lik, gen, 400000, 10, 1.0);
        const double mh = mean_of(d), sh = sd_of(d, mh);
        std::printf("  (poisson: exact mean %.5f sd %.5f | laplace m %.5f v %.5f"
                    " -> old M-step mean bias %.2f sd)\n",
                    ref.mean, ref.sd, q.m, q.v, std::abs(q.m - ref.mean) / ref.sd);
        ok(std::abs(mh - ref.mean) < 0.03 * ref.sd,
           "T1a poisson: |mean - exact| < 0.03 sd", std::abs(mh - ref.mean), 0.03 * ref.sd);
        ok(std::abs(sh / ref.sd - 1.0) < 0.03,
           "T1a poisson: |sd/exact - 1| < 0.03", std::abs(sh / ref.sd - 1.0), 0.03);
        std::sort(d.begin(), d.end());
        for (double p : {0.05, 0.25, 0.5, 0.75, 0.95}) {
            char buf[72];
            std::snprintf(buf, sizeof buf, "T1a poisson: |q%.2f - exact| < 0.05 sd", p);
            const double qe = d[(std::size_t)(p * (d.size() - 1))];
            ok(std::abs(qe - ref.q(p)) < 0.05 * ref.sd, buf, std::abs(qe - ref.q(p)), 0.05 * ref.sd);
        }
    }
    {   // Logistic (Bernoulli) leaf vs grid
        const double sigma_mu = 0.6;
        genbart::lik::logistic_lik lik;
        std::vector<double> ys   = {1, 0, 1, 1, 0, 1, 0, 1, 1, 1};
        std::vector<double> lams = {0.3, -0.5, 0.1, 0.7, -0.2, 0.0, 0.4, -0.6, 0.2, 0.5};
        std::vector<std::size_t> idx(10); for (std::size_t i = 0; i < 10; ++i) idx[i] = i;
        grid_ref ref = grid_conditional(ys, lams, idx, sigma_mu, lik, -4.0, 4.0, 1e-4);
        auto d = run_chain(ys, lams, idx, sigma_mu, lik, gen, 300000, 10, -2.0);
        const double mh = mean_of(d), sh = sd_of(d, mh);
        ok(std::abs(mh - ref.mean) < 0.03 * ref.sd,
           "T1a logistic: |mean - exact| < 0.03 sd", std::abs(mh - ref.mean), 0.03 * ref.sd);
        ok(std::abs(sh / ref.sd - 1.0) < 0.03,
           "T1a logistic: |sd/exact - 1| < 0.03", std::abs(sh / ref.sd - 1.0), 0.03);
    }
    {   // WHOLE-MODEL pure-prior recovery: flat likelihood -> r(x) ~ N(0, T s_mu^2)
        bart_rng::set_seed(777u);
        const std::size_t N = 200, p = 2, T = 20;
        const double sigma_mu = 0.15;
        std::mt19937_64 dgp(3);
        std::uniform_real_distribution<double> u(-1, 1);
        arma::mat X(N, p);
        for (std::size_t i = 0; i < N; ++i) for (std::size_t j = 0; j < p; ++j) X(i, j) = u(dgp);
        arma::vec y(N, arma::fill::zeros);
        flat_lik lik;
        rjmcmc_hypers h; h.sigma_mu = sigma_mu; h.adaptive_sigma_mu = false;
        genbart_model m(X, y, arma::vec(), &lik, h, T);
        const double var_target = (double)T * sigma_mu * sigma_mu;
        for (int b = 0; b < 500; ++b) m.update_step();          // burn
        std::vector<double> r0;                                  // r(x_0) draws
        double acc_m = 0, acc_v = 0; std::size_t nacc = 0;
        for (int s = 0; s < 4000; ++s) {
            m.update_step();
            const double v = m.current_f_train(0);
            r0.push_back(v);
            for (std::size_t i = 0; i < 5; ++i) {                // 5 fixed points
                const double w = m.current_f_train(i * 17);
                acc_m += w; acc_v += w * w; ++nacc;
            }
        }
        const double pm = acc_m / nacc;
        const double pv = acc_v / nacc - pm * pm;
        std::printf("  (pure-prior: pooled mean %.4f, var %.5f vs target %.5f)\n",
                    pm, pv, var_target);
        ok(std::abs(pm) < 0.06 * std::sqrt(var_target),
           "T1a pure-prior: |pooled mean| < 0.06 sd_prior", std::abs(pm),
           0.06 * std::sqrt(var_target));
        ok(std::abs(pv / var_target - 1.0) < 0.15,
           "T1a pure-prior: |var/(T s_mu^2) - 1| < 0.15",
           std::abs(pv / var_target - 1.0), 0.15);
        const double r0m = mean_of(r0), r0s = sd_of(r0, r0m);
        ok(std::abs(r0s / std::sqrt(var_target) - 1.0) < 0.15,
           "T1a pure-prior: r(x0) sd matches prior sd (15%)",
           std::abs(r0s / std::sqrt(var_target) - 1.0), 0.15);
    }

    // T1b -- explicitly silent (validate.md Sec.2 family-silence rule)
    std::printf("================ T1b: FD-gradient -- SILENT by mechanism "
                "(slice M-step uses only log_f; no hand-written gradient) ====\n");

    // =======================================================================
    // T2 -- recovery from known truth (Poisson DGP)
    // =======================================================================
    std::printf("================ T2: recovery from known truth ================\n");
    {
        bart_rng::set_seed(101u);
        const std::size_t N = 400, p = 3, T = 50;
        std::mt19937_64 dgp(11);
        std::uniform_real_distribution<double> u(-1.2, 1.2);
        arma::mat X(N, p);
        arma::vec r_true(N), y(N);
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = 0; j < p; ++j) X(i, j) = u(dgp);
            r_true[i] = 0.8 * std::sin(2 * X(i, 0)) + 0.6 * X(i, 1) - 0.4 * X(i, 2);
            std::poisson_distribution<int> po(std::exp(r_true[i]));
            y[i] = (double)po(dgp);
        }
        genbart::lik::poisson_lik lik;
        rjmcmc_hypers h;                                  // library defaults
        genbart_model m(X, y, arma::vec(), &lik, h, T);
        for (int b = 0; b < 400; ++b) m.update_step();
        arma::vec fbar(N, arma::fill::zeros);
        const int keep = 800;
        for (int s = 0; s < keep; ++s) { m.update_step(); fbar += m.current_f_train(); }
        fbar /= keep;
        const double corr = arma::as_scalar(arma::cor(fbar, r_true));
        const double rmse = std::sqrt(arma::mean(arma::square(fbar - r_true)));
        std::printf("  (T2: corr(f_hat, r_true) = %.4f, RMSE = %.4f)\n", corr, rmse);
        ok(corr > 0.80, "T2 poisson DGP: corr(f_hat, r_true) > 0.80", corr, 0.80);
        ok(rmse < 0.40, "T2 poisson DGP: RMSE(f_hat, r_true) < 0.40", rmse, 0.40);
    }

    // =======================================================================
    // T3 -- cross-chain rank R-hat < 1.01 (two seeds, over-dispersed inits)
    // =======================================================================
    std::printf("================ T3: cross-chain rank R-hat ================\n");
    {
        const std::size_t N = 300, p = 3, T = 50;
        std::mt19937_64 dgp(21);
        std::uniform_real_distribution<double> u(-1.2, 1.2);
        arma::mat X(N, p);
        arma::vec r_true(N), y(N);
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = 0; j < p; ++j) X(i, j) = u(dgp);
            r_true[i] = 0.8 * std::sin(2 * X(i, 0)) + 0.6 * X(i, 1) - 0.4 * X(i, 2);
            std::poisson_distribution<int> po(std::exp(r_true[i]));
            y[i] = (double)po(dgp);
        }
        arma::vec y_shuf = arma::shuffle(y);              // wrong signal for warm start
        const int burn = 800, keep = 1600;
        const std::size_t n_pts = 15;                     // r(x) marginals tracked
        std::vector<std::vector<double> > cA(n_pts), cB(n_pts);

        genbart::lik::poisson_lik likA, likB;
        rjmcmc_hypers h;
        // chain A: seed 1, cold start from stumps (r == 0)
        bart_rng::set_seed(1001u);
        genbart_model mA(X, y, arma::vec(), &likA, h, T);
        for (int b = 0; b < burn; ++b) mA.update_step();
        for (int s = 0; s < keep; ++s) {
            mA.update_step();
            for (std::size_t k = 0; k < n_pts; ++k) cA[k].push_back(mA.current_f_train(k * 19));
        }
        // chain B: seed 2, DELIBERATELY over-dispersed: warm-start the forest
        // on SHUFFLED y (wrong function), then switch to the real y.
        bart_rng::set_seed(2002u);
        genbart_model mB(X, y_shuf, arma::vec(), &likB, h, T);
        for (int w = 0; w < 150; ++w) mB.update_step();
        mB.set_Y(y);
        for (int b = 0; b < burn; ++b) mB.update_step();
        for (int s = 0; s < keep; ++s) {
            mB.update_step();
            for (std::size_t k = 0; k < n_pts; ++k) cB[k].push_back(mB.current_f_train(k * 19));
        }
        double rmax = 0; std::size_t argmax = 0;
        for (std::size_t k = 0; k < n_pts; ++k) {
            const double r = cross_chain_rank_rhat(cA[k], cB[k]);
            if (r > rmax) { rmax = r; argmax = k; }
        }
        std::printf("  (T3: max cross-chain rank R-hat over %zu r(x) marginals"
                    " = %.5f at point %zu)\n", n_pts, rmax, argmax);
        ok(rmax < 1.01, "T3: max cross-chain rank R-hat < 1.01 (MANDATORY)", rmax, 1.01);
    }

    // =======================================================================
    // T4 -- stress: hard regimes + vendored-kernel stateful checks
    // =======================================================================
    std::printf("================ T4: stress + stateful ================\n");
    bart_rng::set_seed(999u);
    {   // huge leaf: n = 5000, conditional sd << w = sigma_mu (shrinkage path)
        const double sigma_mu = 0.5;
        genbart::lik::poisson_lik lik;
        const std::size_t n = 5000;
        std::vector<double> ys(n), lams(n);
        std::vector<std::size_t> idx(n);
        std::mt19937_64 g2(5);
        std::poisson_distribution<int> po(1.3);
        for (std::size_t i = 0; i < n; ++i) { ys[i] = po(g2); lams[i] = 0.2; idx[i] = i; }
        grid_ref ref = grid_conditional(ys, lams, idx, sigma_mu, lik, -1.0, 1.0, 2e-5);
        auto d = run_chain(ys, lams, idx, sigma_mu, lik, gen, 30000, 3, 2.0);
        const double mh = mean_of(d), sh = sd_of(d, mh);
        ok(std::abs(mh - ref.mean) < 0.05 * ref.sd,
           "T4 huge leaf n=5000: |mean - exact| < 0.05 sd", std::abs(mh - ref.mean), 0.05 * ref.sd);
        ok(std::abs(sh / ref.sd - 1.0) < 0.05,
           "T4 huge leaf n=5000: |sd/exact - 1| < 0.05", std::abs(sh / ref.sd - 1.0), 0.05);
    }
    {   // wide prior: sigma_mu = 5, w = 5 vs conditional sd ~ 0.15 (worst-case shrink)
        const double sigma_mu = 5.0;
        genbart::lik::poisson_lik lik;
        const std::size_t n = 50;
        std::vector<double> ys(n), lams(n);
        std::vector<std::size_t> idx(n);
        std::mt19937_64 g2(6);
        std::poisson_distribution<int> po(2.0);
        for (std::size_t i = 0; i < n; ++i) { ys[i] = po(g2); lams[i] = 0.0; idx[i] = i; }
        grid_ref ref = grid_conditional(ys, lams, idx, sigma_mu, lik, -3.0, 3.0, 5e-5);
        auto d = run_chain(ys, lams, idx, sigma_mu, lik, gen, 200000, 10, 4.5);
        const double mh = mean_of(d), sh = sd_of(d, mh);
        ok(std::abs(mh - ref.mean) < 0.03 * ref.sd,
           "T4 wide prior s_mu=5: |mean - exact| < 0.03 sd", std::abs(mh - ref.mean), 0.03 * ref.sd);
        ok(std::abs(sh / ref.sd - 1.0) < 0.03,
           "T4 wide prior s_mu=5: |sd/exact - 1| < 0.03", std::abs(sh / ref.sd - 1.0), 0.03);
    }
    {   // all-zero counts: left-skewed conditional
        const double sigma_mu = 0.8;
        genbart::lik::poisson_lik lik;
        const std::size_t n = 30;
        std::vector<double> ys(n, 0.0), lams(n, 0.0);
        std::vector<std::size_t> idx(n);
        for (std::size_t i = 0; i < n; ++i) idx[i] = i;
        grid_ref ref = grid_conditional(ys, lams, idx, sigma_mu, lik, -6.0, 3.0, 1e-4);
        auto d = run_chain(ys, lams, idx, sigma_mu, lik, gen, 200000, 10, 1.0);
        const double mh = mean_of(d), sh = sd_of(d, mh);
        ok(std::abs(mh - ref.mean) < 0.03 * ref.sd,
           "T4 all-zero counts: |mean - exact| < 0.03 sd", std::abs(mh - ref.mean), 0.03 * ref.sd);
        ok(std::abs(sh / ref.sd - 1.0) < 0.03,
           "T4 all-zero counts: |sd/exact - 1| < 0.03", std::abs(sh / ref.sd - 1.0), 0.03);
    }
    {   // extreme lambda: safe_exp saturation region -- finite draws, no hang
        const double sigma_mu = 0.5;
        genbart::lik::poisson_lik lik;
        std::vector<double> ys   = {1, 0, 2};
        std::vector<double> lams = {30.0, -30.0, 0.0};
        std::vector<std::size_t> idx = {0, 1, 2};
        auto d = run_chain(ys, lams, idx, sigma_mu, lik, gen, 5000, 1, 0.0);
        bool fin = true; for (double x : d) fin = fin && std::isfinite(x);
        ok(fin, "T4 extreme lambda (+/-30): all draws finite, no hang", fin ? 1 : 0, 1);
    }
    {   // vendored-kernel stateful: same-seed determinism (fresh model, re-seed)
        const std::size_t N = 120, p = 2, T = 10;
        std::mt19937_64 dgp(9);
        std::uniform_real_distribution<double> u(-1, 1);
        arma::mat X(N, p);
        arma::vec y(N);
        for (std::size_t i = 0; i < N; ++i) {
            X(i, 0) = u(dgp); X(i, 1) = u(dgp);
            std::poisson_distribution<int> po(std::exp(0.5 * X(i, 0)));
            y[i] = (double)po(dgp);
        }
        genbart::lik::poisson_lik lik1, lik2;
        rjmcmc_hypers h;
        bart_rng::set_seed(31415u);
        genbart_model m1(X, y, arma::vec(), &lik1, h, T);
        for (int s = 0; s < 60; ++s) m1.update_step();
        arma::vec f1 = m1.current_f_train();
        bart_rng::set_seed(31415u);
        genbart_model m2(X, y, arma::vec(), &lik2, h, T);
        for (int s = 0; s < 60; ++s) m2.update_step();
        arma::vec f2 = m2.current_f_train();
        const double dmax = arma::abs(f1 - f2).max();
        ok(dmax == 0.0, "T4 stateful: same-seed determinism (bit-exact)", dmax, 0.0);

        // two-instance state isolation: stepping B must not mutate A's state
        genbart::lik::poisson_lik lik3;
        arma::vec f_before = m1.current_f_train();
        arma::mat Xq = X.rows(0, 19);
        arma::vec pred_before = m1.predict_once(Xq);
        genbart_model mB2(X, y, arma::vec(), &lik3, h, T);
        for (int s = 0; s < 40; ++s) mB2.update_step();
        const double d1 = arma::abs(m1.current_f_train() - f_before).max();
        const double d2 = arma::abs(m1.predict_once(Xq) - pred_before).max();
        ok(d1 == 0.0 && d2 == 0.0,
           "T4 stateful: two-instance isolation (A untouched by B)",
           std::max(d1, d2), 0.0);
    }

    std::printf("\n%s  (%d checks, %d fail)\n",
                fails == 0 ? "[GENBART SLICE LADDER PASS]" : "[GENBART SLICE LADDER FAIL]",
                checks, fails);
    return fails;
}

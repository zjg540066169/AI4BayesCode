// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
// test_dirichlet_gibbs_block.cpp
//
// LIBRARY-LEVEL parity test for AI4BayesCode::dirichlet_gibbs_block.
//
// WHAT IS VERIFIED
// ----------------
// dirichlet_gibbs_block claims that each step() produces an EXACT draw
//
//     theta | ctx  ~  Dirichlet(alpha_post_fn(ctx))
//
// obtained by gamma-normalization (g_k ~ Gamma(alpha_k, 1), theta = g/sum(g)).
// That target is available in closed form, so this test compares the block's
// output directly against the analytic Dirichlet -- the strongest of the
// options available. The weaker fallbacks (invariance of a kernel started at
// its target, exact enumeration, detailed balance) are unnecessary here: the
// block carries no state between steps and no accept/reject move, so its
// draws are iid from a distribution we can write down and integrate exactly.
//
// This covers the MECHANISM, not any one model's alpha_post_fn. Every example
// that uses this block with a textbook conjugate update -- FiniteGaussianMixture
// (Dirichlet(alpha/K + n_k)), HMMGaussian2State (transition rows), HDP -- gets
// its simplex draw from the code exercised here.
//
// The draws being iid (not a Markov chain) is what licenses the plain iid
// standard errors used below; no effective-sample-size correction is needed.
//
// CHECKS
// ------
// Case A -- moderate concentration, alpha = (2, 3, 5, 10), alpha0 = 20:
//   A1. component means      vs  E[theta_k]     = alpha_k / alpha0
//   A2. component variances  vs  Var[theta_k]   = a_k (a0 - a_k) / (a0^2 (a0+1))
//   A3. all 6 pairwise covariances vs Cov[theta_j, theta_k]
//                                    = -a_j a_k / (a0^2 (a0+1))
//       The negative between-component dependence is the part of the JOINT law
//       that componentwise moment checks cannot see, so it is checked
//       explicitly.
//   A4. one-sample Kolmogorov-Smirnov of each marginal against its exact law
//       theta_k ~ Beta(alpha_k, alpha0 - alpha_k). This tests the whole
//       marginal distribution, not just its first two moments -- a sampler
//       with the right mean and variance but the wrong shape fails here.
//   A5. simplex constraint: |sum_k theta_k - 1| at machine precision, and
//       every entry strictly positive.
//
// Case B -- sparse concentration, alpha = (0.5, 0.5, 0.5), alpha0 = 1.5:
//   Same mean / variance / KS checks. The block draws its gammas with
//   std::gamma_distribution, and shapes below 1 take a different branch there
//   than shapes >= 1 (a boost step rather than the Marsaglia-Tsang squeeze),
//   so Case A alone would leave that path untested -- and it is precisely the
//   path the block's header warns about for underflow. Every check in this
//   file is distributional (standard-error bands and a KS statistic), so it
//   does not depend on that distribution's stream being identical across
//   standard libraries.
//
// Case C -- the oracle is re-read from the CURRENT context every step:
//   One block instance, alpha_post_fn = 1 + ctx["counts"]. Draw under
//   counts = (10, 20, 70), then set_context to counts = (70, 20, 10) and draw
//   again; both batches must match their own analytic Dirichlet means. An
//   implementation that cached alpha at construction, or ignored set_context,
//   passes Cases A and B but fails this.
//
// TOLERANCES
// ----------
// Every moment threshold is 4 analytic standard errors for the number of
// draws taken -- no measured or hand-tuned constants.
//   mean:  se = sqrt(mu2 / N)
//   var:   se = sqrt((mu4 - mu2^2) / N)                 (leading term of
//                                                        Var[s^2] at large N)
//   cov:   se = sqrt((E[(X-mx)^2 (Y-my)^2] - cov^2) / N)
// mu2, mu4 and the cross fourth moment are computed EXACTLY from the Dirichlet
// raw-moment formula
//   E[prod_k theta_k^{r_k}] = prod_k rising(alpha_k, r_k) / rising(a0, sum r),
// so no tolerance depends on a simulation. A 4-sigma band has per-check
// false-alarm probability 6.3e-5; with 29 such checks the family-wise rate is
// about 2e-3.
//
// The KS threshold is 1.95 / sqrt(N), the 0.999 quantile of the Kolmogorov
// distribution (K_{0.999} = 1.9495) -- 1e-3 false alarm per marginal, 7e-3
// over the 7 marginals tested.
//
// The simplex tolerance 1e-12 is not statistical: theta is formed by K
// divisions by a common total, so the sum differs from 1 only by O(K * eps)
// ~ 1e-15 of rounding. 1e-12 is a loose machine-precision bound; a real
// normalization bug is off by orders of magnitude more.
//
// Seeds are fixed, so the outcome is deterministic; the false-alarm rates
// above describe the design of the thresholds, not run-to-run flakiness.
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
#include "AI4BayesCode/dirichlet_gibbs_block.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>

#include "../../tests/portable_rng.hpp"   // portable draws: identical on every stdlib
#include <string>
#include <vector>

using AI4BayesCode::block_context;
using AI4BayesCode::dirichlet_gibbs_block;
using AI4BayesCode::dirichlet_gibbs_block_config;

namespace {

// ---- Exact Dirichlet moments ------------------------------------------------

// Rising factorial a (a+1) ... (a+n-1); rising(a, 0) = 1.
double rising(double a, int n) {
    double p = 1.0;
    for (int i = 0; i < n; ++i) p *= (a + static_cast<double>(i));
    return p;
}

// E[ prod_k theta_k^{r_k} ] under Dirichlet(alpha). Exact, no approximation.
double dir_raw_moment(const arma::vec& alpha, const std::vector<int>& r) {
    const double a0 = arma::accu(alpha);
    int R = 0;
    for (int v : r) R += v;
    double num = 1.0;
    for (std::size_t k = 0; k < alpha.n_elem; ++k) num *= rising(alpha[k], r[k]);
    return num / rising(a0, R);
}

// Mean, 2nd and 4th CENTRAL moments of the marginal theta_k
// (which is exactly Beta(alpha_k, a0 - alpha_k)).
void marginal_moments(const arma::vec& alpha, std::size_t k,
                      double& m1, double& mu2, double& mu4) {
    std::vector<int> r(alpha.n_elem, 0);
    r[k] = 1; const double m1r = dir_raw_moment(alpha, r);
    r[k] = 2; const double m2r = dir_raw_moment(alpha, r);
    r[k] = 3; const double m3r = dir_raw_moment(alpha, r);
    r[k] = 4; const double m4r = dir_raw_moment(alpha, r);
    m1  = m1r;
    mu2 = m2r - m1r * m1r;
    mu4 = m4r - 4.0 * m1r * m3r + 6.0 * m1r * m1r * m2r
          - 3.0 * m1r * m1r * m1r * m1r;
}

// Cov(theta_j, theta_k) and E[(theta_j - m_j)^2 (theta_k - m_k)^2], both exact.
void pair_moments(const arma::vec& alpha, std::size_t j, std::size_t k,
                  double& cov, double& cross4) {
    auto M = [&](int rj, int rk) {
        std::vector<int> r(alpha.n_elem, 0);
        r[j] = rj; r[k] = rk;
        return dir_raw_moment(alpha, r);
    };
    const double a = M(1, 0);
    const double b = M(0, 1);
    cov = M(1, 1) - a * b;
    // (X-a)^2 (Y-b)^2 expanded into raw moments.
    cross4 = M(2, 2)
           - 2.0 * b * M(2, 1) + b * b * M(2, 0)
           - 2.0 * a * M(1, 2) + 4.0 * a * b * M(1, 1) - 2.0 * a * b * b * M(1, 0)
           + a * a * M(0, 2) - 2.0 * a * a * b * M(0, 1) + a * a * b * b;
}

// ---- One-sample KS statistic against Beta(a, b) -----------------------------

double ks_stat_beta(std::vector<double> x, double a, double b) {
    std::sort(x.begin(), x.end());
    const double n = static_cast<double>(x.size());
    double d = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const double F  = R::pbeta(x[i], a, b, 1, 0);
        const double up = static_cast<double>(i + 1) / n - F;
        const double lo = F - static_cast<double>(i) / n;
        d = std::max(d, std::max(up, lo));
    }
    return d;
}

// ---- Draw N samples from a freshly built block ------------------------------

std::vector<std::vector<double>>
draw_block(const arma::vec& alpha, std::size_t n_draws, std::uint64_t seed,
           double& max_sum_err, double& min_entry) {
    const std::size_t K = alpha.n_elem;

    dirichlet_gibbs_block_config cfg;
    cfg.name           = "theta_test";
    cfg.n_categories   = K;
    cfg.initial_values = arma::vec(K, arma::fill::value(1.0 / static_cast<double>(K)));
    cfg.alpha_post_fn  = [alpha](const block_context& /*ctx*/) { return alpha; };
    dirichlet_gibbs_block blk(std::move(cfg));

    block_context ctx;                 // alpha_post_fn ignores it here
    blk.set_context(ctx);

    std::vector<std::vector<double>> s(K);
    for (std::size_t k = 0; k < K; ++k) s[k].reserve(n_draws);

    std::mt19937_64 rng(seed);
    max_sum_err = 0.0;
    min_entry   = 1.0;
    for (std::size_t i = 0; i < n_draws; ++i) {
        blk.step(rng);
        const arma::vec& v = blk.current();
        double sum = 0.0;
        for (std::size_t k = 0; k < K; ++k) {
            s[k].push_back(v[k]);
            sum += v[k];
            if (!(v[k] > 0.0) || !std::isfinite(v[k])) min_entry = -1.0;
            else min_entry = std::min(min_entry, v[k]);
        }
        max_sum_err = std::max(max_sum_err, std::abs(sum - 1.0));
    }
    return s;
}

double sample_mean(const std::vector<double>& x) {
    double m = 0.0;
    for (double v : x) m += v;
    return m / static_cast<double>(x.size());
}

double sample_var(const std::vector<double>& x, double m) {
    double s = 0.0;
    for (double v : x) { const double d = v - m; s += d * d; }
    return s / static_cast<double>(x.size() - 1);
}

// Worst |estimate - analytic| / analytic_se over all components and,
// optionally, all pairs. Also returns the worst KS statistic.
struct case_summary {
    double max_z_mean = 0.0;
    double max_z_var  = 0.0;
    double max_z_cov  = 0.0;
    double max_ks     = 0.0;
    double max_sum_err = 0.0;
    double min_entry  = 1.0;
    arma::vec samp_mean, exp_mean, samp_var, exp_var;
};

case_summary summarize(const arma::vec& alpha,
                       const std::vector<std::vector<double>>& s,
                       bool do_cov) {
    const std::size_t K = alpha.n_elem;
    const double N = static_cast<double>(s[0].size());
    const double a0 = arma::accu(alpha);

    case_summary out;
    out.samp_mean.set_size(K); out.exp_mean.set_size(K);
    out.samp_var.set_size(K);  out.exp_var.set_size(K);

    std::vector<double> mhat(K);
    for (std::size_t k = 0; k < K; ++k) {
        double m1, mu2, mu4;
        marginal_moments(alpha, k, m1, mu2, mu4);

        const double mh = sample_mean(s[k]);
        const double vh = sample_var(s[k], mh);
        mhat[k] = mh;

        const double se_mean = std::sqrt(mu2 / N);
        const double se_var  = std::sqrt(std::max(mu4 - mu2 * mu2, 0.0) / N);

        out.max_z_mean = std::max(out.max_z_mean, std::abs(mh - m1)  / se_mean);
        out.max_z_var  = std::max(out.max_z_var,  std::abs(vh - mu2) / se_var);

        out.samp_mean[k] = mh; out.exp_mean[k] = m1;
        out.samp_var[k]  = vh; out.exp_var[k]  = mu2;

        // Marginal law of theta_k is exactly Beta(alpha_k, a0 - alpha_k).
        out.max_ks = std::max(out.max_ks,
                              ks_stat_beta(s[k], alpha[k], a0 - alpha[k]));
    }

    if (do_cov) {
        for (std::size_t j = 0; j < K; ++j) {
            for (std::size_t k = j + 1; k < K; ++k) {
                double cov_true, cross4;
                pair_moments(alpha, j, k, cov_true, cross4);
                double c = 0.0;
                for (std::size_t i = 0; i < s[j].size(); ++i)
                    c += (s[j][i] - mhat[j]) * (s[k][i] - mhat[k]);
                c /= (N - 1.0);
                const double se_cov =
                    std::sqrt(std::max(cross4 - cov_true * cov_true, 0.0) / N);
                out.max_z_cov = std::max(out.max_z_cov,
                                         std::abs(c - cov_true) / se_cov);
            }
        }
    }
    return out;
}

} // namespace

// [[Rcpp::export]]
Rcpp::List test_dirichlet_gibbs_block() {
    const std::size_t N_A = 40000;   // Case A draws
    const std::size_t N_B = 40000;   // Case B draws
    const std::size_t N_C = 20000;   // Case C draws per context

    const double Z_TOL  = 4.0;       // 4 analytic standard errors

    // ---- Case A: alpha = (2, 3, 5, 10) ----------------------------------
    const arma::vec alpha_a = arma::vec({2.0, 3.0, 5.0, 10.0});
    double sum_err_a = 0.0, min_a = 1.0;
    const std::vector<std::vector<double>> sa =
        draw_block(alpha_a, N_A, 20260818ULL, sum_err_a, min_a);
    case_summary A = summarize(alpha_a, sa, /*do_cov=*/true);
    A.max_sum_err = sum_err_a; A.min_entry = min_a;

    const double ks_crit_a = 1.9495 / std::sqrt(static_cast<double>(N_A));

    const bool pass_mean_a = (A.max_z_mean < Z_TOL);
    const bool pass_var_a  = (A.max_z_var  < Z_TOL);
    const bool pass_cov_a  = (A.max_z_cov  < Z_TOL);
    const bool pass_ks_a   = (A.max_ks     < ks_crit_a);
    const bool pass_simplex_a = (A.max_sum_err < 1e-12) && (A.min_entry > 0.0);

    // ---- Case B: sparse alpha = (0.5, 0.5, 0.5) -------------------------
    const arma::vec alpha_b = arma::vec({0.5, 0.5, 0.5});
    double sum_err_b = 0.0, min_b = 1.0;
    const std::vector<std::vector<double>> sb =
        draw_block(alpha_b, N_B, 777001ULL, sum_err_b, min_b);
    case_summary B = summarize(alpha_b, sb, /*do_cov=*/false);
    B.max_sum_err = sum_err_b; B.min_entry = min_b;

    const double ks_crit_b = 1.9495 / std::sqrt(static_cast<double>(N_B));

    const bool pass_mean_b = (B.max_z_mean < Z_TOL);
    const bool pass_var_b  = (B.max_z_var  < Z_TOL);
    const bool pass_ks_b   = (B.max_ks     < ks_crit_b);
    const bool pass_simplex_b = (B.max_sum_err < 1e-12) && (B.min_entry > 0.0);

    // ---- Case C: the oracle must be re-read from the CURRENT context ----
    // One block, two contexts. alpha_post = 1 + counts.
    double max_z_mean_c = 0.0;
    {
        dirichlet_gibbs_block_config cfg;
        cfg.name           = "theta_ctx";
        cfg.n_categories   = 3;
        cfg.initial_values = arma::vec({1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0});
        cfg.alpha_post_fn  = [](const block_context& ctx) -> arma::vec {
            return 1.0 + ctx.at("counts");
        };
        dirichlet_gibbs_block blk(std::move(cfg));

        const arma::vec counts1({10.0, 20.0, 70.0});
        const arma::vec counts2({70.0, 20.0, 10.0});
        std::mt19937_64 rng(31337ULL);

        for (int phase = 0; phase < 2; ++phase) {
            const arma::vec& cnt = (phase == 0) ? counts1 : counts2;
            block_context ctx;
            ctx["counts"] = cnt;
            blk.set_context(ctx);

            const arma::vec alpha_c = 1.0 + cnt;
            std::vector<std::vector<double>> sc(3);
            for (std::size_t k = 0; k < 3; ++k) sc[k].reserve(N_C);
            for (std::size_t i = 0; i < N_C; ++i) {
                blk.step(rng);
                const arma::vec& v = blk.current();
                for (std::size_t k = 0; k < 3; ++k) sc[k].push_back(v[k]);
            }
            for (std::size_t k = 0; k < 3; ++k) {
                double m1, mu2, mu4;
                marginal_moments(alpha_c, k, m1, mu2, mu4);
                const double mh = sample_mean(sc[k]);
                const double se = std::sqrt(mu2 / static_cast<double>(N_C));
                max_z_mean_c = std::max(max_z_mean_c, std::abs(mh - m1) / se);
            }
        }
    }
    const bool pass_context = (max_z_mean_c < Z_TOL);

    const bool all_pass =
        pass_mean_a && pass_var_a && pass_cov_a && pass_ks_a && pass_simplex_a &&
        pass_mean_b && pass_var_b && pass_ks_b && pass_simplex_b &&
        pass_context;

    return Rcpp::List::create(
        Rcpp::Named("all_pass")        = all_pass,
        Rcpp::Named("pass_mean_a")     = pass_mean_a,
        Rcpp::Named("pass_var_a")      = pass_var_a,
        Rcpp::Named("pass_cov_a")      = pass_cov_a,
        Rcpp::Named("pass_ks_a")       = pass_ks_a,
        Rcpp::Named("pass_simplex_a")  = pass_simplex_a,
        Rcpp::Named("pass_mean_b")     = pass_mean_b,
        Rcpp::Named("pass_var_b")      = pass_var_b,
        Rcpp::Named("pass_ks_b")       = pass_ks_b,
        Rcpp::Named("pass_simplex_b")  = pass_simplex_b,
        Rcpp::Named("pass_context")    = pass_context,
        Rcpp::Named("max_z_mean_a")    = A.max_z_mean,
        Rcpp::Named("max_z_var_a")     = A.max_z_var,
        Rcpp::Named("max_z_cov_a")     = A.max_z_cov,
        Rcpp::Named("max_ks_a")        = A.max_ks,
        Rcpp::Named("ks_crit_a")       = ks_crit_a,
        Rcpp::Named("max_z_mean_b")    = B.max_z_mean,
        Rcpp::Named("max_z_var_b")     = B.max_z_var,
        Rcpp::Named("max_ks_b")        = B.max_ks,
        Rcpp::Named("ks_crit_b")       = ks_crit_b,
        Rcpp::Named("max_z_mean_c")    = max_z_mean_c,
        Rcpp::Named("z_tol")           = Z_TOL,
        Rcpp::Named("samp_mean_a")     = A.samp_mean,
        Rcpp::Named("exp_mean_a")      = A.exp_mean,
        Rcpp::Named("samp_var_a")      = A.samp_var,
        Rcpp::Named("exp_var_a")       = A.exp_var,
        Rcpp::Named("max_sum_err_a")   = A.max_sum_err,
        Rcpp::Named("max_sum_err_b")   = B.max_sum_err,
        Rcpp::Named("min_entry_a")     = A.min_entry,
        Rcpp::Named("min_entry_b")     = B.min_entry,
        Rcpp::Named("n_draws")         = Rcpp::IntegerVector::create(
                                            static_cast<int>(N_A),
                                            static_cast<int>(N_B),
                                            static_cast<int>(N_C)));
}

// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
// test_stick_breaking_block.cpp
//
// LIBRARY-LEVEL parity test for AI4BayesCode::stick_breaking_block.
//
// What the block does
// -------------------
// Given user-supplied Beta-parameter oracles a_fn(k, counts, ctx) and
// b_fn(k, counts, ctx), it draws
//
//     V_k ~ Beta(a_k, b_k)   independently, k = 0, ..., K_trunc-2
//     V_{K_trunc-1} = 1                                (Ishwaran-James truncation)
//     pi_k = V_k * prod_{j<k} (1 - V_j)
//
// and exposes pi (and optionally V) as named outputs. The draw does NOT
// depend on the block's own current state, so this is an independence
// sampler: the distribution it produces IS its target. That makes an
// invariance / detailed-balance argument unnecessary -- we can compare
// the sampled law directly to a closed-form one. Exact enumeration does
// not apply either, since pi is continuous.
//
// Two analytic targets are used, and they check different things.
//
// TARGET 1 -- the Connor & Mosimann (1969) stick-breaking identity
// ----------------------------------------------------------------
// If the stick parameters are set to
//
//     a_k = alpha_k,   b_k = sum_{j>k} alpha_j,     A = sum_j alpha_j
//
// then the resulting pi is EXACTLY Dirichlet(alpha_1, ..., alpha_K). So the
// whole JOINT law is known in closed form, not just the marginals:
//
//     E[pi_k]        = alpha_k / A
//     Var[pi_k]      = alpha_k (A - alpha_k) / (A^2 (A + 1))
//     Cov[pi_j,pi_k] = -alpha_j alpha_k / (A^2 (A + 1))     (j != k)
//
// The covariance row is the load-bearing part. K marginal mean/variance
// checks would also pass for a block that drew the K entries independently
// and normalised; only the off-diagonal terms pin down the stick-breaking
// dependence structure that the block header explicitly warns is different
// from a plain Dirichlet fit. Here K = 4, alpha = (2, 3, 1, 4), A = 10.
//
// TARGET 2 -- a Dirichlet-process posterior conditional driven by counts
// ----------------------------------------------------------------------
// Target 1's a_k, b_k do not depend on the counts vector at all, so it
// would still pass if the block never read counts_key or mis-indexed it.
// Target 2 closes that hole: with concentration alpha read from the
// context and cluster counts n read from counts_key,
//
//     a_k = 1 + n_k,   b_k = alpha + sum_{j>k} n_j        (Sethuraman 1994)
//
// so a dropped counts read, a shifted k, or a reversed tail sum all move
// the target. Here K = 6, alpha = 2, n = (10, 5, 3, 1, 0, 0).
//
// pi_k is now a product of independent Beta variables and has no named
// marginal, but every raw moment is still exact because the sticks are
// independent:
//
//     E[pi_k^m] = E[V_k^m] * prod_{j<k} E[(1 - V_j)^m]
//     E[V^m]    = (a)_m / (a+b)_m      for V ~ Beta(a, b)
//     E[(1-V)^m]= (b)_m / (a+b)_m      since 1 - V ~ Beta(b, a)
//
// with (x)_m the rising factorial. Mean and variance are compared against
// these; the moments are built from the Beta parameters, NOT from anything
// the block returns.
//
// TARGET 3 -- structural identities that must hold on EVERY draw
// ---------------------------------------------------------------
//   sum_k pi_k == 1   (exact by telescoping once V_{K-1} = 1)
//   pi_k > 0
//   pi reconstructed from the exposed V vector equals the exposed pi
//   set_current(pi) -> exposed V -> reconstruct == pi   (inverse stick map)
//
// Tolerances
// ----------
// Every moment threshold is 5 x the ANALYTIC standard error at N = 200000
// draws -- no relative-error convention, because the SE is computable here:
//
//   se(mean_hat) = sqrt(mu2 / N)
//   se(var_hat)  = sqrt((mu4 - mu2^2) / N)
//   se(cov_hat)  = sqrt((E[(X-mx)^2 (Y-my)^2] - cov^2) / N)
//
// mu2 and mu4 are the exact central moments of each target (Beta moments
// for Target 1, stick-product moments for Target 2); the bivariate fourth
// moment for the covariance SE comes from the Dirichlet joint raw moment
// E[pi_j^p pi_k^q] = (alpha_j)_p (alpha_k)_q / (A)_{p+q}.
//
// There are 26 moment checks in all (Target 1: 4 means, 4 variances, 6
// covariances; Target 2: 6 means, 6 variances). Five SE gives a per-check
// false-alarm rate of 5.7e-7, so the family-wise rate is about 1.5e-5 and
// the fixed seed is not doing any work. Measured over 10 seeds during
// authoring (20260818, 1, 7, 99, 123456, 2024, 31337, 555, 8675309, 42)
// the worst |z| over all 26 checks was 2.85, consistent with the
// asymptotic SEs above. max_abs_z is returned so a failure reports how far
// off it was.
//
// The structural thresholds are 1e-12, an absolute bound: sum(pi) is
// renormalised inside step() and the stick reconstruction is a product of
// K <= 6 factors, so rounding is O(K * eps) ~ 1e-15. 1e-12 leaves three
// orders of headroom and still fails on any real algebra error. Observed
// max deviations at N = 200000 are 4.4e-16 (sum) and 2.2e-16 (stick
// reconstruction).
//
// Power (mutants run during authoring; each was rejected, so no check here
// is vacuous)
// ------------------------------------------------------------------------
//   a_fn made to ignore the counts vector             -> max |z| = 4415
//   b_fn tail summed over j >= k instead of j > k     -> max |z| =  319
//   covariances scored against the independence null  -> max |z| =  198
// The last line is the one that matters for the joint-law claim: pi drawn
// with the right marginals but independent entries would be rejected at
// roughly 200 standard errors.
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
#include "AI4BayesCode/stick_breaking_block.hpp"

#include <cmath>
#include <random>
#include <vector>

using AI4BayesCode::block_context;
using AI4BayesCode::stick_breaking_block;
using AI4BayesCode::stick_breaking_block_config;

namespace {

// Rising factorial (x)_m = x (x+1) ... (x+m-1), with (x)_0 = 1.
double rising_(double x, int m) {
    double out = 1.0;
    for (int r = 0; r < m; ++r) out *= (x + static_cast<double>(r));
    return out;
}

// E[V^m] for V ~ Beta(a, b).
double beta_raw_(double a, double b, int m) {
    return rising_(a, m) / rising_(a + b, m);
}

// Central moments 2 and 4 from raw moments raw[1..4].
void central_(const double raw[5], double& mu, double& mu2, double& mu4) {
    mu  = raw[1];
    mu2 = raw[2] - mu * mu;
    mu4 = raw[4] - 4.0 * raw[3] * mu + 6.0 * raw[2] * mu * mu
        - 3.0 * mu * mu * mu * mu;
}

// Dirichlet joint raw moment E[pi_j^p pi_k^q] for j != k.
double dir_raw2_(double aj, double ak, double A, int p, int q) {
    return rising_(aj, p) * rising_(ak, q) / rising_(A, p + q);
}

// Reconstruct pi from a stick vector V: pi_k = V_k prod_{j<k} (1 - V_j).
arma::vec pi_from_v_(const arma::vec& v) {
    arma::vec pi(v.n_elem);
    double rem = 1.0;
    for (arma::uword k = 0; k < v.n_elem; ++k) {
        pi[k] = v[k] * rem;
        rem  *= (1.0 - v[k]);
    }
    return pi;
}

}  // namespace

// [[Rcpp::export]]
Rcpp::List test_stick_breaking_block() {
    const std::size_t N_DRAWS = 200000;
    const double N_SE = 5.0;        // 5 analytic standard errors (see header)
    const double STRUCT_TOL = 1e-12;

    // Structural diagnostics accumulated over BOTH targets.
    double max_sum_err = 0.0;
    double min_pi      = 1.0;
    double max_v_recon = 0.0;

    bool all_pass = true;

    // Worst standardised deviation over every moment check, so a failure
    // reports HOW far off it was rather than just "false".
    double max_abs_z = 0.0;
    auto chk = [&](double sample, double truth, double se) {
        const double z = (sample - truth) / se;
        if (std::abs(z) > max_abs_z) max_abs_z = std::abs(z);
        if (std::abs(z) > N_SE) all_pass = false;
        return z;
    };

    // ======================================================================
    // TARGET 1: Connor-Mosimann -- sticks tuned so that pi ~ Dirichlet(alpha)
    // ======================================================================
    const std::size_t K1 = 4;
    const arma::vec alpha1({2.0, 3.0, 1.0, 4.0});
    const double A1 = arma::sum(alpha1);   // 10

    {
        stick_breaking_block_config cfg;
        cfg.name       = "pi_dir";
        cfg.K_trunc    = K1;
        cfg.counts_key = "cluster_counts";
        cfg.v_name     = "v_dir";
        cfg.initial_pi = arma::vec(K1, arma::fill::ones) / static_cast<double>(K1);
        // a_k = alpha_k, b_k = sum_{j>k} alpha_j.
        cfg.a_fn = [alpha1](std::size_t k, const arma::vec& /*counts*/,
                            const block_context& /*ctx*/) {
            return alpha1[k];
        };
        cfg.b_fn = [alpha1](std::size_t k, const arma::vec& /*counts*/,
                            const block_context& /*ctx*/) {
            double tail = 0.0;
            for (arma::uword j = k + 1; j < alpha1.n_elem; ++j) tail += alpha1[j];
            return tail;
        };
        stick_breaking_block blk(std::move(cfg));

        block_context ctx;
        // The block requires counts of length K_trunc even though Target 1's
        // stick parameters ignore them.
        ctx["cluster_counts"] = arma::vec(K1, arma::fill::zeros);
        blk.set_context(ctx);

        std::mt19937_64 rng(20260818);
        std::vector<double> S1(K1, 0.0);
        std::vector<double> S2(K1 * K1, 0.0);   // raw cross-products
        for (std::size_t it = 0; it < N_DRAWS; ++it) {
            blk.step(rng);
            const arma::vec& pi = blk.current();
            double s = 0.0;
            for (std::size_t k = 0; k < K1; ++k) {
                s += pi[k];
                if (pi[k] < min_pi) min_pi = pi[k];
                S1[k] += pi[k];
                for (std::size_t l = 0; l < K1; ++l) S2[k * K1 + l] += pi[k] * pi[l];
            }
            const double serr = std::abs(s - 1.0);
            if (serr > max_sum_err) max_sum_err = serr;

            // V-consistency on a 1-in-200 subsample (the map build is the
            // expensive part; the identity is deterministic, so a subsample
            // is enough to catch a wrong reconstruction).
            if (it % 200 == 0) {
                auto outs = blk.current_named_outputs();
                const arma::vec recon = pi_from_v_(outs.at("v_dir"));
                const double d = arma::max(arma::abs(recon - outs.at("pi_dir")));
                if (d > max_v_recon) max_v_recon = d;
            }
        }

        const double n = static_cast<double>(N_DRAWS);
        for (std::size_t k = 0; k < K1; ++k) {
            const double mk = S1[k] / n;
            const double vk = (S2[k * K1 + k] - n * mk * mk) / (n - 1.0);

            // Analytic Dirichlet marginal: pi_k ~ Beta(alpha_k, A - alpha_k).
            const double a = alpha1[k], b = A1 - alpha1[k];
            double raw[5];
            raw[0] = 1.0;
            for (int m = 1; m <= 4; ++m) raw[m] = beta_raw_(a, b, m);
            double mu, mu2, mu4;
            central_(raw, mu, mu2, mu4);

            chk(mk, mu,  std::sqrt(mu2 / n));
            chk(vk, mu2, std::sqrt((mu4 - mu2 * mu2) / n));
        }
        // Off-diagonal covariances -- the joint-law check.
        for (std::size_t k = 0; k < K1; ++k) {
            for (std::size_t l = k + 1; l < K1; ++l) {
                const double mk = S1[k] / n, ml = S1[l] / n;
                const double ck = (S2[k * K1 + l] - n * mk * ml) / (n - 1.0);

                const double aj = alpha1[k], ak = alpha1[l];
                const double cov_true = -aj * ak / (A1 * A1 * (A1 + 1.0));
                const double mx = aj / A1, my = ak / A1;
                // E[(X-mx)^2 (Y-my)^2] from Dirichlet joint raw moments.
                const double E22 = dir_raw2_(aj, ak, A1, 2, 2);
                const double E21 = dir_raw2_(aj, ak, A1, 2, 1);
                const double E12 = dir_raw2_(aj, ak, A1, 1, 2);
                const double E11 = dir_raw2_(aj, ak, A1, 1, 1);
                const double E20 = dir_raw2_(aj, ak, A1, 2, 0);
                const double E02 = dir_raw2_(aj, ak, A1, 0, 2);
                const double E10 = mx, E01 = my;
                const double m22 = E22 - 2.0 * my * E21 + my * my * E20
                                 - 2.0 * mx * E12 + 4.0 * mx * my * E11
                                 - 2.0 * mx * my * my * E10
                                 + mx * mx * E02 - 2.0 * mx * mx * my * E01
                                 + mx * mx * my * my;
                chk(ck, cov_true, std::sqrt((m22 - cov_true * cov_true) / n));
            }
        }
    }

    // ======================================================================
    // TARGET 2: DP posterior conditional, a_k and b_k driven by counts
    // ======================================================================
    const std::size_t K2 = 6;
    const arma::vec counts2({10.0, 5.0, 3.0, 1.0, 0.0, 0.0});
    const double dp_alpha = 2.0;

    std::vector<double> t2_mean(K2, 0.0), t2_var(K2, 0.0);
    std::vector<double> t2_exp_mean(K2, 0.0), t2_exp_var(K2, 0.0);
    std::vector<double> t2_z_mean(K2, 0.0), t2_z_var(K2, 0.0);

    {
        stick_breaking_block_config cfg;
        cfg.name       = "pi_dp";
        cfg.K_trunc    = K2;
        cfg.counts_key = "n_k";
        cfg.v_name     = "v_dp";
        cfg.initial_pi = arma::vec(K2, arma::fill::ones) / static_cast<double>(K2);
        // Sethuraman DP sticks: a_k = 1 + n_k, b_k = alpha + sum_{j>k} n_j.
        // Both read the counts ARGUMENT (not a capture) so a missing or
        // mis-indexed counts read shows up in the moments.
        cfg.a_fn = [](std::size_t k, const arma::vec& counts,
                      const block_context& /*ctx*/) {
            return 1.0 + counts[k];
        };
        cfg.b_fn = [](std::size_t k, const arma::vec& counts,
                      const block_context& ctx) {
            const double alpha = ctx.at("dp_alpha")[0];
            double tail = 0.0;
            for (arma::uword j = k + 1; j < counts.n_elem; ++j) tail += counts[j];
            return alpha + tail;
        };
        stick_breaking_block blk(std::move(cfg));

        block_context ctx;
        ctx["n_k"]      = counts2;
        ctx["dp_alpha"] = arma::vec({dp_alpha});
        blk.set_context(ctx);

        std::mt19937_64 rng(20260818);
        std::vector<double> S1(K2, 0.0), S2(K2, 0.0);
        for (std::size_t it = 0; it < N_DRAWS; ++it) {
            blk.step(rng);
            const arma::vec& pi = blk.current();
            double s = 0.0;
            for (std::size_t k = 0; k < K2; ++k) {
                s += pi[k];
                if (pi[k] < min_pi) min_pi = pi[k];
                S1[k] += pi[k];
                S2[k] += pi[k] * pi[k];
            }
            const double serr = std::abs(s - 1.0);
            if (serr > max_sum_err) max_sum_err = serr;

            if (it % 200 == 0) {
                auto outs = blk.current_named_outputs();
                const arma::vec recon = pi_from_v_(outs.at("v_dp"));
                const double d = arma::max(arma::abs(recon - outs.at("pi_dp")));
                if (d > max_v_recon) max_v_recon = d;
            }
        }

        // Exact stick parameters, rebuilt here independently of the block.
        std::vector<double> a2(K2, 0.0), b2(K2, 0.0);
        for (std::size_t k = 0; k + 1 < K2; ++k) {
            a2[k] = 1.0 + counts2[k];
            double tail = 0.0;
            for (std::size_t j = k + 1; j < K2; ++j) tail += counts2[j];
            b2[k] = dp_alpha + tail;
        }

        const double n = static_cast<double>(N_DRAWS);
        for (std::size_t k = 0; k < K2; ++k) {
            // E[pi_k^m] = E[V_k^m] prod_{j<k} E[(1-V_j)^m]; the last stick is
            // deterministically 1 so its E[V^m] = 1.
            double raw[5];
            raw[0] = 1.0;
            for (int m = 1; m <= 4; ++m) {
                double t = (k + 1 < K2) ? beta_raw_(a2[k], b2[k], m) : 1.0;
                for (std::size_t j = 0; j < k; ++j) t *= beta_raw_(b2[j], a2[j], m);
                raw[m] = t;
            }
            double mu, mu2, mu4;
            central_(raw, mu, mu2, mu4);

            const double mk = S1[k] / n;
            const double vk = (S2[k] - n * mk * mk) / (n - 1.0);
            const double se_mean = std::sqrt(mu2 / n);
            const double se_var  = std::sqrt((mu4 - mu2 * mu2) / n);

            t2_mean[k] = mk;      t2_var[k] = vk;
            t2_exp_mean[k] = mu;  t2_exp_var[k] = mu2;
            t2_z_mean[k] = chk(mk, mu,  se_mean);
            t2_z_var[k]  = chk(vk, mu2, se_var);
        }

        // ---- set_current round trip: pi -> V -> pi ------------------------
        arma::vec pi_set({0.40, 0.25, 0.15, 0.10, 0.06, 0.04});
        pi_set /= arma::sum(pi_set);
        blk.set_current(pi_set);
        auto outs = blk.current_named_outputs();
        const arma::vec recon = pi_from_v_(outs.at("v_dp"));
        const double d = arma::max(arma::abs(recon - pi_set));
        if (d > max_v_recon) max_v_recon = d;
    }

    const bool pass_struct = (max_sum_err <= STRUCT_TOL) &&
                             (min_pi > 0.0) &&
                             (max_v_recon <= STRUCT_TOL);
    all_pass = all_pass && pass_struct;

    return Rcpp::List::create(
        Rcpp::Named("all_pass")      = all_pass,
        Rcpp::Named("pass_struct")   = pass_struct,
        Rcpp::Named("max_sum_err")   = max_sum_err,
        Rcpp::Named("min_pi")        = min_pi,
        Rcpp::Named("max_v_recon")   = max_v_recon,
        Rcpp::Named("struct_tol")    = STRUCT_TOL,
        Rcpp::Named("max_abs_z")     = max_abs_z,
        Rcpp::Named("dp_mean")       = Rcpp::NumericVector(t2_mean.begin(), t2_mean.end()),
        Rcpp::Named("dp_exp_mean")   = Rcpp::NumericVector(t2_exp_mean.begin(), t2_exp_mean.end()),
        Rcpp::Named("dp_var")        = Rcpp::NumericVector(t2_var.begin(), t2_var.end()),
        Rcpp::Named("dp_exp_var")    = Rcpp::NumericVector(t2_exp_var.begin(), t2_exp_var.end()),
        Rcpp::Named("dp_z_mean")     = Rcpp::NumericVector(t2_z_mean.begin(), t2_z_mean.end()),
        Rcpp::Named("dp_z_var")      = Rcpp::NumericVector(t2_z_var.begin(), t2_z_var.end()),
        Rcpp::Named("n_draws")       = static_cast<int>(N_DRAWS),
        Rcpp::Named("n_se_tol")      = N_SE);
}

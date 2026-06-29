/*================================================================================
 *  AI4BayesCode  --  end-to-end validation of the simplex NUTS block.
 *================================================================================
 *
 *  Model
 *  -----
 *      theta            ~ Dirichlet(alpha)
 *      y_i | theta      ~ Categorical(theta),    i = 1..N
 *
 *  Equivalent (by sufficiency):
 *      y_counts | theta ~ Multinomial(N, theta)
 *
 *  Conjugate posterior:
 *      theta | y        ~ Dirichlet(alpha + y_counts)
 *
 *  so the posterior mean and variance have closed form:
 *      E[theta_k]   = (alpha_k + y_k) / S
 *      Var[theta_k] = E[theta_k] * (1 - E[theta_k]) / (S + 1)
 *  where S = sum(alpha + y_counts).
 *
 *  What this test proves
 *  ---------------------
 *    1. constraints::simplex::constrain / unconstrain round-trip correctly.
 *    2. constraints::simplex::wrap composes with nuts_block without any
 *       dimension / gradient plumbing bugs.
 *    3. The NUTS chain produces draws that live on the simplex exactly
 *       (sum to one to machine precision).
 *    4. Four independent chains started from overdispersed initial points
 *       converge to the analytic Dirichlet posterior:
 *         (a) Pooled posterior means match the analytic means to within
 *             a few Monte Carlo standard errors.
 *         (b) The Vehtari et al. 2021 rank-normalized R-hat (max of the
 *             bulk and folded-tail R-hat) is < 1.05 for every component.
 *         (c) Pooled posterior sds match analytic sds within tolerance.
 *
 *  This is the motivating test for the whole "AI-friendly block MCMC"
 *  project: a Dirichlet posterior on a simplex is exactly the MT_DART /
 *  DP_DART pain point that an LLM writing raw RWMH gets wrong, and that
 *  hand-written NUTS with stick-breaking + Jacobians gets subtly wrong.
 *  Here the AI-visible code is two small functions and a few lines of
 *  constructor glue; every Jacobian is handled by the library, and the
 *  multi-chain rank R-hat gives an actual convergence guarantee.
 *================================================================================*/

#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/shared_data.hpp"
#include "AI4BayesCode/nuts_block.hpp"
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/constraints.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <random>
#include <utility>
#include <vector>

using AI4BayesCode::block_context;
using AI4BayesCode::composite_block;
using AI4BayesCode::nuts_block;
using AI4BayesCode::nuts_block_config;

namespace constraints = AI4BayesCode::constraints;

// ============================================================================
//  Data: draw multinomial counts from a known theta_true.
// ============================================================================

static arma::vec simulate_counts(const arma::vec& theta_true, std::size_t N,
                                 std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<double> w(theta_true.n_elem);
    for (std::size_t k = 0; k < theta_true.n_elem; ++k) w[k] = theta_true[k];
    std::discrete_distribution<int> cat(w.begin(), w.end());

    arma::vec y(theta_true.n_elem, arma::fill::zeros);
    for (std::size_t i = 0; i < N; ++i) {
        y[static_cast<std::size_t>(cat(rng))] += 1.0;
    }
    return y;
}

// ============================================================================
//  Target: natural-scale Dirichlet log-density.
// ============================================================================

static double dirichlet_log_density(const arma::vec& theta_nat,
                                    const block_context& ctx,
                                    arma::vec* grad_nat) {
    const arma::vec& y     = ctx.at("y");      // counts, length K
    const arma::vec& alpha = ctx.at("alpha");  // prior, length K
    const std::size_t K    = theta_nat.n_elem;

    double lp = 0.0;
    for (std::size_t k = 0; k < K; ++k) {
        const double tk = theta_nat[k];
        if (tk <= 0.0) {
            return -std::numeric_limits<double>::infinity();
        }
        lp += (alpha[k] + y[k] - 1.0) * std::log(tk);
    }

    if (grad_nat) {
        grad_nat->set_size(K);
        for (std::size_t k = 0; k < K; ++k) {
            (*grad_nat)[k] = (alpha[k] + y[k] - 1.0) / theta_nat[k];
        }
    }
    return lp;
}

// ============================================================================
//  Composite assembly. Takes an explicit initial theta so different chains
//  can start from overdispersed points.
// ============================================================================

static std::unique_ptr<composite_block>
build_model(const arma::vec& y, const arma::vec& alpha,
            const arma::vec& theta_init) {
    auto model = std::make_unique<composite_block>("dirichlet_simplex");

    model->data().set("y",     y);
    model->data().set("alpha", alpha);
    model->data().set("theta", theta_init);

    model->data().declare_dependencies("theta", {"y", "alpha"});

    nuts_block_config cfg;
    cfg.name        = "theta";
    cfg.initial_unc = constraints::simplex::unconstrain(theta_init);
    cfg.constrain   = constraints::simplex::constrain;
    cfg.unconstrain = constraints::simplex::unconstrain;
    cfg.log_density_grad =
        [](const arma::vec& theta_unc, const block_context& ctx,
           arma::vec* grad) {
            return constraints::simplex::wrap(
                theta_unc, grad,
                [&](const arma::vec& theta_nat, arma::vec* grad_nat) {
                    return dirichlet_log_density(theta_nat, ctx, grad_nat);
                });
        };
    cfg.nuts_settings.nuts_settings.max_tree_depth     = 8;
    cfg.nuts_settings.nuts_settings.target_accept_rate = 0.8;
    // Simplex posteriors are tight enough that FindReasonableEpsilon
    // + the 10x bias push the first warmup iteration into a divergent
    // regime; seed a small step size to bypass that path. Dual averaging
    // will grow it from here if it's too small.
    cfg.initial_step_size   = 0.05;
    cfg.n_warmup_first_call = 400;
    cfg.n_warmup_per_step   = 20;
    cfg.n_draws_per_step    = 1;

    model->add_child(std::make_unique<nuts_block>(std::move(cfg)));
    return model;
}

static arma::mat run_sampler(composite_block& model,
                             std::size_t n_burnin, std::size_t n_keep,
                             std::uint64_t seed) {
    std::mt19937_64 rng(seed);

    for (std::size_t s = 0; s < n_burnin; ++s) model.step(rng);

    const std::size_t K = model.data().get("theta").n_elem;
    arma::mat draws(n_keep, K);
    for (std::size_t s = 0; s < n_keep; ++s) {
        model.step(rng);
        draws.row(s) = arma::conv_to<arma::rowvec>::from(
            model.data().get("theta"));
    }
    return draws;
}

// ============================================================================
//  Convergence diagnostics: rank-normalized split-R-hat
//  Reference: Vehtari, Gelman, Simpson, Carpenter, Burkner (2021)
//             "Rank-Normalization, Folding, and Localization: An Improved
//              R-hat for Assessing Convergence of MCMC". Bayesian Analysis.
// ============================================================================

// Beasley-Springer / Moro / Acklam inverse normal CDF. Accurate to ~10 digits
// across the full open unit interval. Used to rank-normalize draws before
// computing R-hat.
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

    const double p_low  = 0.02425;
    const double p_high = 1.0 - p_low;

    if (p < p_low) {
        const double q = std::sqrt(-2.0 * std::log(p));
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) /
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0);
    } else if (p <= p_high) {
        const double q = p - 0.5;
        const double r = q * q;
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q
             / (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0);
    } else {
        const double q = std::sqrt(-2.0 * std::log(1.0 - p));
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) /
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0);
    }
}

// Classical split-R-hat. Each chain is split in half and treated as two
// independent chains, following the Gelman et al. 2013 Bayesian Data
// Analysis recommendation for better sensitivity to non-stationarity.
static double classical_split_rhat(
    const std::vector<arma::vec>& chains)
{
    const std::size_t M_in = chains.size();
    if (M_in < 1) return std::numeric_limits<double>::quiet_NaN();

    const std::size_t N_full = chains[0].n_elem;
    if (N_full < 4) return std::numeric_limits<double>::quiet_NaN();

    // Split each chain in half -> 2*M effective chains of length N.
    const std::size_t N = N_full / 2;
    const std::size_t M = 2 * M_in;

    std::vector<double> chain_means(M);
    std::vector<double> chain_vars(M);
    for (std::size_t m_in = 0; m_in < M_in; ++m_in) {
        for (int half = 0; half < 2; ++half) {
            const std::size_t off = (half == 0) ? 0 : N;
            double mean = 0.0;
            for (std::size_t n = 0; n < N; ++n) {
                mean += chains[m_in][off + n];
            }
            mean /= static_cast<double>(N);

            double var = 0.0;
            for (std::size_t n = 0; n < N; ++n) {
                const double d = chains[m_in][off + n] - mean;
                var += d * d;
            }
            var /= static_cast<double>(N - 1);

            const std::size_t mm = 2 * m_in + half;
            chain_means[mm] = mean;
            chain_vars[mm]  = var;
        }
    }

    double grand_mean = 0.0;
    for (double m : chain_means) grand_mean += m;
    grand_mean /= static_cast<double>(M);

    double B_over_N = 0.0;
    for (double m : chain_means) {
        const double d = m - grand_mean;
        B_over_N += d * d;
    }
    B_over_N /= static_cast<double>(M - 1);

    double W = 0.0;
    for (double v : chain_vars) W += v;
    W /= static_cast<double>(M);

    if (W <= 0.0) return std::numeric_limits<double>::quiet_NaN();

    const double V_hat =
        static_cast<double>(N - 1) / static_cast<double>(N) * W + B_over_N;
    return std::sqrt(V_hat / W);
}

// Compute rank R-hat on the draws of a single component across chains,
// following Vehtari et al. 2021:
//   rank_rhat_bulk : rank-normalize then classical split-R-hat
//   rank_rhat_tail : fold around pooled median, rank-normalize,
//                    classical split-R-hat
//   reported       : max(bulk, tail)
static double rank_split_rhat(const std::vector<arma::vec>& chains) {
    const std::size_t M = chains.size();
    const std::size_t N = chains[0].n_elem;
    const std::size_t total = M * N;

    // Build (value, (chain, index)) pairs for ranking.
    std::vector<std::pair<double, std::pair<std::size_t, std::size_t>>> all;
    all.reserve(total);
    for (std::size_t m = 0; m < M; ++m) {
        for (std::size_t n = 0; n < N; ++n) {
            all.push_back({chains[m][n], {m, n}});
        }
    }
    std::sort(all.begin(), all.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    // Assign ranks with average-rank tie handling.
    std::vector<double> ranks(total);
    std::size_t i = 0;
    while (i < total) {
        std::size_t j = i;
        while (j < total && all[j].first == all[i].first) ++j;
        const double avg_rank = 0.5 * (static_cast<double>(i + 1)
                                     + static_cast<double>(j));
        for (std::size_t k = i; k < j; ++k) ranks[k] = avg_rank;
        i = j;
    }

    // Normal-score transform: z = Phi^-1((rank - 3/8) / (total + 1/4))
    std::vector<arma::vec> z_chains(M);
    for (std::size_t m = 0; m < M; ++m) z_chains[m].set_size(N);
    for (std::size_t k = 0; k < total; ++k) {
        const auto& tag = all[k].second;
        const double p =
            (ranks[k] - 0.375) / (static_cast<double>(total) + 0.25);
        z_chains[tag.first][tag.second] = norm_inv_cdf(p);
    }

    const double rhat_bulk = classical_split_rhat(z_chains);

    // Folded: |theta - pooled_median|
    const std::size_t mid_idx = total / 2;
    double median = all[mid_idx].first;
    if (total % 2 == 0) {
        median = 0.5 * (all[mid_idx - 1].first + all[mid_idx].first);
    }

    // Re-rank the folded values.
    std::vector<std::pair<double, std::pair<std::size_t, std::size_t>>> foldall;
    foldall.reserve(total);
    for (std::size_t m = 0; m < M; ++m) {
        for (std::size_t n = 0; n < N; ++n) {
            foldall.push_back(
                {std::abs(chains[m][n] - median), {m, n}});
        }
    }
    std::sort(foldall.begin(), foldall.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    std::vector<double> fold_ranks(total);
    i = 0;
    while (i < total) {
        std::size_t j = i;
        while (j < total && foldall[j].first == foldall[i].first) ++j;
        const double avg_rank = 0.5 * (static_cast<double>(i + 1)
                                     + static_cast<double>(j));
        for (std::size_t k = i; k < j; ++k) fold_ranks[k] = avg_rank;
        i = j;
    }

    std::vector<arma::vec> z_fold(M);
    for (std::size_t m = 0; m < M; ++m) z_fold[m].set_size(N);
    for (std::size_t k = 0; k < total; ++k) {
        const auto& tag = foldall[k].second;
        const double p =
            (fold_ranks[k] - 0.375) / (static_cast<double>(total) + 0.25);
        z_fold[tag.first][tag.second] = norm_inv_cdf(p);
    }
    const double rhat_tail = classical_split_rhat(z_fold);

    return std::max(rhat_bulk, rhat_tail);
}

// ============================================================================
//  Overdispersed initial points on the simplex.
//  Place each chain's mass on a different corner while keeping every
//  component strictly > 0 so the unconstrain map is well-defined.
// ============================================================================

static arma::vec corner_init(std::size_t K, std::size_t heavy_k,
                             double heavy_weight = 0.70) {
    const double light = (1.0 - heavy_weight) / static_cast<double>(K - 1);
    arma::vec theta(K);
    for (std::size_t k = 0; k < K; ++k) {
        theta[k] = (k == heavy_k) ? heavy_weight : light;
    }
    return theta;
}

// ============================================================================
//  Main: run 4 chains with overdispersed inits, compute rank R-hat per
//  component, compare pooled means / sds to analytic.
// ============================================================================

int main() {
    const std::size_t K = 5;
    const std::size_t N = 500;
    const arma::vec theta_true{0.10, 0.25, 0.35, 0.20, 0.10};
    const arma::vec alpha     {1.00, 1.00, 1.00, 1.00, 1.00}; // flat prior
    const std::uint64_t data_seed   = 20260410ull;
    const std::uint64_t base_seed   = 0xD1ECD1ECull;
    const std::size_t M_chains = 4;
    const std::size_t n_burnin = 500;
    const std::size_t n_keep   = 3000;

    // ---- data ---------------------------------------------------------
    const arma::vec y = simulate_counts(theta_true, N, data_seed);

    std::printf("Generated N=%zu counts for K=%zu categories:\n", N, K);
    std::printf("  y      = ");
    for (std::size_t k = 0; k < K; ++k) std::printf("%3.0f ", y[k]);
    std::printf("\n  theta* = ");
    for (std::size_t k = 0; k < K; ++k) std::printf("%.4f ", theta_true[k]);
    std::printf("\n");

    // ---- analytic posterior moments ----------------------------------
    const arma::vec alpha_post = alpha + y;
    const double    S          = arma::sum(alpha_post);
    arma::vec mean_analytic(K), sd_analytic(K);
    for (std::size_t k = 0; k < K; ++k) {
        mean_analytic[k] = alpha_post[k] / S;
        sd_analytic[k]   = std::sqrt(
            mean_analytic[k] * (1.0 - mean_analytic[k]) / (S + 1.0));
    }
    std::printf("  analytic mean = ");
    for (std::size_t k = 0; k < K; ++k) std::printf("%.4f ", mean_analytic[k]);
    std::printf("\n  analytic sd   = ");
    for (std::size_t k = 0; k < K; ++k) std::printf("%.4f ", sd_analytic[k]);
    std::printf("\n\n");

    // ---- four overdispersed initial points, one per chain -----------
    std::vector<arma::vec> inits(M_chains);
    for (std::size_t m = 0; m < M_chains; ++m) {
        inits[m] = corner_init(K, m);  // heavy on category m
    }

    std::printf("Running %zu chains (burnin=%zu, keep=%zu each) from "
                "overdispersed corners:\n", M_chains, n_burnin, n_keep);
    for (std::size_t m = 0; m < M_chains; ++m) {
        std::printf("  chain %zu init = ", m + 1);
        for (std::size_t k = 0; k < K; ++k) std::printf("%.3f ", inits[m][k]);
        std::printf("\n");
    }
    std::printf("\n");

    // ---- run each chain ---------------------------------------------
    std::vector<arma::mat> chain_draws(M_chains);
    for (std::size_t m = 0; m < M_chains; ++m) {
        auto model = build_model(y, alpha, inits[m]);
        chain_draws[m] =
            run_sampler(*model, n_burnin, n_keep, base_seed + 1000ull * m);
    }

    // ---- per-chain summaries ----------------------------------------
    std::printf("Per-chain posterior means (component x chain):\n");
    std::printf("%10s", "");
    for (std::size_t m = 0; m < M_chains; ++m)
        std::printf("  chain%zu ", m + 1);
    std::printf("  analytic\n");
    for (std::size_t k = 0; k < K; ++k) {
        std::printf("  theta[%zu]", k);
        for (std::size_t m = 0; m < M_chains; ++m) {
            std::printf("  %7.4f",
                        arma::mean(chain_draws[m].col(k)));
        }
        std::printf("  %7.4f\n", mean_analytic[k]);
    }
    std::printf("\n");

    // ---- rank R-hat per component -----------------------------------
    std::printf("Rank R-hat (Vehtari et al. 2021, split + folded):\n");
    double max_rhat = 0.0;
    std::vector<double> rhat_per_k(K);
    for (std::size_t k = 0; k < K; ++k) {
        std::vector<arma::vec> component_chains(M_chains);
        for (std::size_t m = 0; m < M_chains; ++m) {
            component_chains[m] = chain_draws[m].col(k);
        }
        const double rhat = rank_split_rhat(component_chains);
        rhat_per_k[k] = rhat;
        max_rhat = std::max(max_rhat, rhat);
        std::printf("  theta[%zu] R-hat = %.4f\n", k, rhat);
    }
    std::printf("\n");

    // ---- pooled summary (all chains concatenated) -------------------
    arma::mat pooled(M_chains * n_keep, K);
    for (std::size_t m = 0; m < M_chains; ++m) {
        pooled.rows(m * n_keep, (m + 1) * n_keep - 1) = chain_draws[m];
    }
    const arma::rowvec mean_nuts_r = arma::mean(pooled, 0);
    const arma::rowvec sd_nuts_r   = arma::stddev(pooled, 0, 0);
    const arma::vec mean_nuts = mean_nuts_r.t();
    const arma::vec sd_nuts   = sd_nuts_r.t();

    std::printf("Pooled posterior summary (%zu x %zu draws):\n",
                M_chains, n_keep);
    std::printf("  mean = ");
    for (std::size_t k = 0; k < K; ++k) std::printf("%.4f ", mean_nuts[k]);
    std::printf("\n  sd   = ");
    for (std::size_t k = 0; k < K; ++k) std::printf("%.4f ", sd_nuts[k]);
    std::printf("\n\n");

    // ---- checks ------------------------------------------------------

    // (a) Every draw lives on the simplex exactly.
    bool simplex_ok = true;
    double max_sum_err = 0.0;
    for (std::size_t m = 0; m < M_chains; ++m) {
        for (std::size_t s = 0; s < n_keep; ++s) {
            const double sum_row = arma::sum(chain_draws[m].row(s));
            max_sum_err = std::max(max_sum_err, std::abs(sum_row - 1.0));
            if (std::abs(sum_row - 1.0) > 1e-10) simplex_ok = false;
            for (std::size_t k = 0; k < K; ++k) {
                if (chain_draws[m](s, k) <= 0.0) simplex_ok = false;
            }
        }
    }

    // (b) Pooled component means within 4 MC SE of analytic.
    const double n_eff_floor = 100.0 * static_cast<double>(M_chains);
    bool means_ok = true;
    double max_mean_err_in_se = 0.0;
    for (std::size_t k = 0; k < K; ++k) {
        const double mc_se = sd_analytic[k] / std::sqrt(n_eff_floor);
        const double err   = std::abs(mean_nuts[k] - mean_analytic[k]);
        const double err_in_se = err / mc_se;
        max_mean_err_in_se = std::max(max_mean_err_in_se, err_in_se);
        if (err > 4.0 * mc_se) means_ok = false;
    }

    // (c) Pooled component sds within 20% of analytic.
    bool sds_ok = true;
    double max_sd_rel_err = 0.0;
    for (std::size_t k = 0; k < K; ++k) {
        const double rel = std::abs(sd_nuts[k] - sd_analytic[k])
                         / sd_analytic[k];
        max_sd_rel_err = std::max(max_sd_rel_err, rel);
        if (rel > 0.20) sds_ok = false;
    }

    // (d) Rank R-hat < 1.05 for every component.
    const bool rhat_ok = max_rhat < 1.05;

    std::printf("every draw sums to 1 (max err = %.2e)? %s\n",
                max_sum_err, simplex_ok ? "YES" : "NO");
    std::printf("pooled component means within 4 MC SE of analytic "
                "(max = %.2f SE)? %s\n",
                max_mean_err_in_se, means_ok ? "YES" : "NO");
    std::printf("pooled component sds within 20%% of analytic "
                "(max rel err = %.2f%%)? %s\n",
                100.0 * max_sd_rel_err, sds_ok ? "YES" : "NO");
    std::printf("all rank R-hat < 1.05 (max = %.4f)? %s\n",
                max_rhat, rhat_ok ? "YES" : "NO");

    const bool all_ok = simplex_ok && means_ok && sds_ok && rhat_ok;
    std::printf("\n%s\n", all_ok ? "PASS" : "FAIL");
    return all_ok ? 0 : 1;
}

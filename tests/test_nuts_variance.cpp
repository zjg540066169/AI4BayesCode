/*
 * test_nuts_variance.cpp
 *
 * Verify NUTS produces correct posterior mean AND variance on a conjugate
 * Gaussian target across multiple seeds and dimensions.
 *
 * Target: y_i ~ N(mu, sigma^2_known), mu ~ N(0, prior_var)
 * Posterior: mu | y ~ N(post_mean, post_var)
 */

#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/shared_data.hpp"
#include "AI4BayesCode/nuts_block.hpp"
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/constraints.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using AI4BayesCode::block_context;
using AI4BayesCode::composite_block;
using AI4BayesCode::nuts_block;
using AI4BayesCode::nuts_block_config;
namespace constraints = AI4BayesCode::constraints;

namespace {

// 1D Gaussian posterior log-density
double mu_log_density_1d(const arma::vec& theta_nat,
                         const block_context& ctx,
                         arma::vec* grad_nat) {
    const double mu = theta_nat[0];
    const arma::vec& y = ctx.at("y");
    const double sigma2 = ctx.at("sigma2")[0];
    const double prior_var = ctx.at("prior_var")[0];
    const double N = static_cast<double>(y.n_elem);

    double sum_y = arma::sum(y);
    double sum_sq = 0.0;
    for (std::size_t i = 0; i < y.n_elem; ++i) {
        double r = y[i] - mu;
        sum_sq += r * r;
    }

    double lp = -0.5 * sum_sq / sigma2 - 0.5 * mu * mu / prior_var;
    if (grad_nat) {
        grad_nat->set_size(1);
        (*grad_nat)[0] = (sum_y - mu * N) / sigma2 - mu / prior_var;
    }
    return lp;
}

// Multi-D diagonal Gaussian: independent components
// mu_j ~ N(0, prior_var), y_{ij} ~ N(mu_j, sigma2)
double mu_log_density_nd(const arma::vec& theta_nat,
                         const block_context& ctx,
                         arma::vec* grad_nat) {
    const arma::vec& sum_y = ctx.at("sum_y");     // length D
    const double N = ctx.at("N_obs")[0];
    const double sigma2 = ctx.at("sigma2")[0];
    const double prior_var = ctx.at("prior_var")[0];
    const std::size_t D = theta_nat.n_elem;

    double lp = 0.0;
    if (grad_nat) grad_nat->set_size(D);
    for (std::size_t j = 0; j < D; ++j) {
        double mu_j = theta_nat[j];
        lp += -0.5 * N * mu_j * mu_j / sigma2
              + mu_j * sum_y[j] / sigma2
              - 0.5 * mu_j * mu_j / prior_var;
        if (grad_nat) {
            (*grad_nat)[j] = (sum_y[j] - N * mu_j) / sigma2
                             - mu_j / prior_var;
        }
    }
    return lp;
}

struct TestResult {
    int seed;
    int dim;
    double var_ratio;
    double mean_err_sd;
    bool pass;
};

TestResult run_1d_test(int data_seed, int mcmc_seed) {
    const int N = 50;
    const double sigma2 = 4.0;
    const double prior_var = 100.0;

    std::mt19937_64 data_rng(data_seed);
    std::normal_distribution<double> norm(0.0, 1.0);

    arma::vec y(N);
    for (int i = 0; i < N; ++i) y[i] = 3.0 + 2.0 * norm(data_rng);

    double sum_y = arma::sum(y);
    double post_var = 1.0 / (N / sigma2 + 1.0 / prior_var);
    double post_mean = post_var * (sum_y / sigma2);
    double post_sd = std::sqrt(post_var);

    auto comp = std::make_unique<composite_block>("test");
    comp->data().set("y", y);
    comp->data().set("sigma2", arma::vec{sigma2});
    comp->data().set("prior_var", arma::vec{prior_var});
    comp->data().set("mu", arma::vec{0.0});
    comp->data().declare_dependencies("mu", {"y", "sigma2", "prior_var"});

    nuts_block_config cfg;
    cfg.name = "mu";
    cfg.initial_unc = arma::vec{0.0};
    cfg.log_density_grad =
        [](const arma::vec& u, const block_context& c, arma::vec* g) {
            return constraints::real::wrap(u, g,
                [&](const arma::vec& n, arma::vec* gn) {
                    return mu_log_density_1d(n, c, gn);
                });
        };
    cfg.n_warmup_first_call = 2000;
    cfg.n_warmup_per_step = 0;
    comp->add_child(std::make_unique<nuts_block>(std::move(cfg)));

    std::mt19937_64 rng(mcmc_seed);
    for (int i = 0; i < 5000; ++i) comp->step(rng);

    double s1 = 0, s2 = 0;
    const int n_keep = 50000;
    for (int i = 0; i < n_keep; ++i) {
        comp->step(rng);
        double mu = comp->data().get("mu")[0];
        s1 += mu;
        s2 += mu * mu;
    }
    double sm = s1 / n_keep;
    double sv = s2 / n_keep - sm * sm;

    TestResult r;
    r.seed = data_seed;
    r.dim = 1;
    r.var_ratio = sv / post_var;
    r.mean_err_sd = std::abs(sm - post_mean) / post_sd;
    r.pass = std::abs(r.var_ratio - 1.0) < 0.05 && r.mean_err_sd < 0.1;
    return r;
}

TestResult run_nd_test(int D, int data_seed, int mcmc_seed) {
    const int N = 100;
    const double sigma2 = 4.0;
    const double prior_var = 100.0;

    std::mt19937_64 data_rng(data_seed);
    std::normal_distribution<double> norm(0.0, 1.0);

    arma::vec sum_y(D, arma::fill::zeros);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < D; ++j) {
            sum_y[j] += 3.0 + 2.0 * norm(data_rng);
        }
    }

    double post_var = 1.0 / (N / sigma2 + 1.0 / prior_var);
    arma::vec post_mean = post_var * (sum_y / sigma2);
    double post_sd = std::sqrt(post_var);

    auto comp = std::make_unique<composite_block>("test");
    comp->data().set("sum_y", sum_y);
    comp->data().set("N_obs", arma::vec{static_cast<double>(N)});
    comp->data().set("sigma2", arma::vec{sigma2});
    comp->data().set("prior_var", arma::vec{prior_var});
    comp->data().set("mu", arma::vec(D, arma::fill::zeros));
    comp->data().declare_dependencies("mu",
        {"sum_y", "N_obs", "sigma2", "prior_var"});

    nuts_block_config cfg;
    cfg.name = "mu";
    cfg.initial_unc = arma::vec(D, arma::fill::zeros);
    cfg.log_density_grad =
        [](const arma::vec& u, const block_context& c, arma::vec* g) {
            return constraints::real::wrap(u, g,
                [&](const arma::vec& n, arma::vec* gn) {
                    return mu_log_density_nd(n, c, gn);
                });
        };
    cfg.n_warmup_first_call = 2000;
    cfg.n_warmup_per_step = 0;
    comp->add_child(std::make_unique<nuts_block>(std::move(cfg)));

    std::mt19937_64 rng(mcmc_seed);
    for (int i = 0; i < 5000; ++i) comp->step(rng);

    arma::vec s1(D, arma::fill::zeros);
    arma::vec s2(D, arma::fill::zeros);
    const int n_keep = 50000;
    for (int i = 0; i < n_keep; ++i) {
        comp->step(rng);
        const arma::vec& mu = comp->data().get("mu");
        s1 += mu;
        s2 += mu % mu;  // element-wise square
    }
    arma::vec sm = s1 / n_keep;
    arma::vec sv = s2 / n_keep - sm % sm;

    // Average variance ratio across dimensions
    double avg_var_ratio = arma::mean(sv) / post_var;
    double max_mean_err = arma::max(arma::abs(sm - post_mean)) / post_sd;

    TestResult r;
    r.seed = data_seed;
    r.dim = D;
    r.var_ratio = avg_var_ratio;
    r.mean_err_sd = max_mean_err;
    r.pass = std::abs(r.var_ratio - 1.0) < 0.05 && r.mean_err_sd < 0.15;
    return r;
}

} // namespace

int main() {
    std::printf("=== NUTS variance calibration test ===\n\n");

    std::vector<TestResult> results;

    // 1D tests with different seeds
    std::printf("--- 1D tests (5 seeds) ---\n");
    for (int seed : {12345, 23456, 34567, 45678, 56789}) {
        auto r = run_1d_test(seed, seed + 1000);
        std::printf("  seed=%d: var_ratio=%.4f  mean_err=%.4f SD  %s\n",
                    r.seed, r.var_ratio, r.mean_err_sd,
                    r.pass ? "OK" : "FAIL");
        results.push_back(r);
    }

    // Multi-D tests (including D=1 to isolate code path vs geometry)
    std::printf("\n--- Multi-D tests (using nd density function) ---\n");
    for (int D : {1, 2, 5, 10, 20}) {
        auto r = run_nd_test(D, 99999, 88888 + D);
        std::printf("  D=%2d: avg_var_ratio=%.4f  max_mean_err=%.4f SD  %s\n",
                    r.dim, r.var_ratio, r.mean_err_sd,
                    r.pass ? "OK" : "FAIL");
        results.push_back(r);
    }

    // Summary
    int n_pass = 0;
    for (const auto& r : results) if (r.pass) n_pass++;
    std::printf("\n%d / %d tests passed.\n", n_pass,
                static_cast<int>(results.size()));

    bool all_pass = (n_pass == static_cast<int>(results.size()));
    std::printf("\n%s\n", all_pass ? "PASS" : "FAIL");
    return all_pass ? 0 : 1;
}

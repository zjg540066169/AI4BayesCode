// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
//  HMMGaussian2State.cpp
//
//  2-state Hidden Markov Model with Gaussian emissions. Demonstrates the
//  hmm_block (T10) forward-filter backward-sample machinery.
//
//  Model
//  -----
//      z_1         ~ Categorical(pi)                   (initial)
//      z_t | z_{t-1} = k  ~ Categorical(A_{k,:})      (transition)
//      y_t | z_t = k  ~ N(mu_k, sigma^2)               (emission)
//
//  For this example, the transition matrix A, initial pi, emission means
//  (mu_0, mu_1), and emission sigma are FIXED at construction. Only the
//  latent state sequence z_1:T is sampled by MCMC via hmm_block's FFBS.
//
//  This is the minimal demo for hmm_block. For a full Bayesian HMM where
//  A / pi / mu / sigma are all sampled, add sibling blocks:
//    - A rows  : dirichlet_gibbs_block per row
//    - pi      : dirichlet_gibbs_block
//    - mu_k    : nuts_block each (or a single joint_nuts_block over {mu_0, mu_1})
//    - sigma   : nuts_block with Jeffreys prior
//    - z       : hmm_block (this file)
//
//  JUSTIFICATION (Check #16): Exception 1 from codegen.md §2b — z is
//  DISCRETE (state sequence over a finite alphabet); NUTS cannot target
//  discrete measures. hmm_block is the library-blessed forward-backward
//  sampler. Check #15 parity test at
//    tests_autodiff/test_hmm_block.cpp
//  verifies FFBS marginals against analytical Baum-Welch smoothing to
//  max_abs_err < 0.2% (10k draws).
// ============================================================================

// Frontend-INDEPENDENT example: pure standalone C++ (NO Rcpp, NO pybind).
// Compile + run directly: it has an int main() that simulates an HMM from a
// known state sequence, fits the hmm_block via FFBS, and checks that the
// posterior-mode state path recovers the true path. No R / Python binding is
// built or required.

#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif

#include <armadillo>

#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/shared_data.hpp"
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/hmm_block.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>

using AI4BayesCode::block_context;
using AI4BayesCode::composite_block;
using AI4BayesCode::hmm_block;
using AI4BayesCode::hmm_block_config;

class HMMGaussian2State {
public:
    HMMGaussian2State(const arma::vec& y,
                      const arma::vec& A_flat_row_major,  // length 4 (2x2)
                      const arma::vec& pi_init,           // length 2
                      const arma::vec& mu,                // length 2
                      double sigma,
                      int rng_seed,
                      bool keep_history = false)
        : rng_(rng_seed == 0
                   ? std::mt19937_64{std::random_device{}()}
                   : std::mt19937_64{static_cast<std::uint64_t>(rng_seed)}),
          predict_rng_(rng_seed == 0
                   ? std::mt19937_64{std::random_device{}()}
                   : std::mt19937_64{static_cast<std::uint64_t>(rng_seed)
                                     ^ 0x9E3779B97F4A7C15ULL}),
          impl_(std::make_unique<composite_block>("HMMGaussian2State")),
          keep_history_(keep_history)
    {
        if (y.n_elem < 2) throw std::runtime_error("y length must be >= 2");
        if (A_flat_row_major.n_elem != 4)
            throw std::runtime_error("A must be length 4 (row-major 2x2)");
        if (pi_init.n_elem != 2) throw std::runtime_error("pi must be length 2");
        if (mu.n_elem != 2)      throw std::runtime_error("mu must be length 2");
        if (!(sigma > 0.0))      throw std::runtime_error("sigma must be > 0");
        // Validate row sums = 1 on A.
        if (std::abs((A_flat_row_major[0] + A_flat_row_major[1]) - 1.0) > 1e-8 ||
            std::abs((A_flat_row_major[2] + A_flat_row_major[3]) - 1.0) > 1e-8) {
            throw std::runtime_error("A rows must sum to 1");
        }
        if (std::abs((pi_init[0] + pi_init[1]) - 1.0) > 1e-8) {
            throw std::runtime_error("pi must sum to 1");
        }
        T_ = y.n_elem;

        impl_->data().set("y",  y);
        impl_->data().set("A",  A_flat_row_major);
        impl_->data().set("pi", pi_init);
        impl_->data().set("mu", mu);
        impl_->data().set("sigma", arma::vec{sigma});

        // Initial z: alternate 0/1.
        arma::vec z_init(T_);
        for (std::size_t t = 0; t < T_; ++t) z_init[t] = static_cast<double>(t % 2);
        impl_->data().set("z", z_init);

        impl_->data().declare_dependencies("z", {"y", "A", "pi", "mu", "sigma"});
        // Predict DAG: y_rep[t] = mu[z[t]] + sigma*epsilon
        // All three (z, mu, sigma) are direct generative parents of y_rep.
        impl_->data().declare_predict_edges("z",     {"y_rep"});
        impl_->data().declare_predict_edges("mu",    {"y_rep"});
        impl_->data().declare_predict_edges("sigma", {"y_rep"});

        // ---- Generative-DAG context (VIZ-ONLY; predict_at BFS never
        //      reads context_edges_). z_1 ~ Categorical(pi);
        //      z_t | z_{t-1} ~ Categorical(A_{z_{t-1},:}): the initial
        //      distribution pi and transition matrix A are z's
        //      generative parents. Drawn faded by plot_dag.
        impl_->data().declare_context_edges("pi", {"z"});
        impl_->data().declare_context_edges("A",  {"z"});
        impl_->data().set("y_rep", arma::vec(T_, arma::fill::zeros));
        impl_->data().register_stochastic_refresher(
            "y_rep",
            [T = T_](const AI4BayesCode::shared_data_t& d,
                     std::mt19937_64& rng) {
                const arma::vec& z_cur = d.get("z");
                const arma::vec& mu    = d.get("mu");
                const double sigma     = d.get("sigma")[0];
                std::normal_distribution<double> nd(0.0, 1.0);
                arma::vec y_rep(T);
                for (std::size_t t = 0; t < T; ++t) {
                    const std::size_t k =
                        static_cast<std::size_t>(z_cur[t]);
                    y_rep[t] = mu[k] + sigma * nd(rng);
                }
                return y_rep;
            });

        // hmm_block with Gaussian emission.
        hmm_block_config cfg;
        cfg.name = "z";
        cfg.T = T_;
        cfg.K = 2;
        cfg.A_key = "A";
        cfg.pi_key = "pi";
        cfg.emission_logp =
            [](std::size_t t, std::size_t k, const block_context& ctx)
            -> double {
                const arma::vec& y  = ctx.at("y");
                const arma::vec& mu = ctx.at("mu");
                const double sigma  = ctx.at("sigma")[0];
                const double r      = y[t] - mu[k];
                return -0.5 * std::log(2.0 * M_PI)
                       - std::log(sigma)
                       - 0.5 * (r * r) / (sigma * sigma);
            };
        cfg.initial_z = z_init;
        impl_->add_child(std::make_unique<hmm_block>(std::move(cfg)));

        if (keep_history_) impl_->set_keep_history(true);
    }

    // ---- Canonical neutral-typed interface ----

    void step(int n_steps) {
        if (n_steps < 0) throw std::runtime_error("n_steps must be >= 0");
        for (int i = 0; i < n_steps; ++i) impl_->step(rng_);
    }

    AI4BayesCode::state_map get_current() const {
        AI4BayesCode::state_map out;
        out["z"] = impl_->data().get("z");
        return out;
    }

    void set_current(const AI4BayesCode::state_map& params) {
        auto* z_blk = dynamic_cast<hmm_block*>(&impl_->child(0));
        auto it_z = params.find("z");
        if (it_z != params.end()) {
            const arma::vec& znew = it_z->second;
            if (znew.n_elem != T_)
                throw std::runtime_error("set_current: z length must equal T");
            z_blk->set_current(znew);
            impl_->data().set("z", znew);
        }
        auto it_y = params.find("y");
        if (it_y != params.end()) {
            const arma::vec& y_new = it_y->second;
            if (y_new.n_elem != T_)
                throw std::runtime_error("y length must equal T");
            impl_->data().set("y", y_new);
        }
    }

    AI4BayesCode::dag_info get_dag() const { return impl_->get_dag(); }
    AI4BayesCode::history_map get_history() const { return impl_->get_history(); }

private:
    std::mt19937_64                  rng_;
    mutable std::mt19937_64          predict_rng_;
    std::unique_ptr<composite_block> impl_;
    bool                             keep_history_ = false;
    std::size_t                      T_ = 0;
};

//==============================================================================
//  Standalone FRONTEND-INDEPENDENT demo (pure C++; no Rcpp, no pybind).
//
//  Simulates a 2-state Gaussian HMM from a KNOWN transition matrix, initial
//  distribution, emission means and sd, AND a known latent state path z*.
//  Then fits the hmm_block (A / pi / mu / sigma FIXED at truth, exactly as the
//  model intends — only z is sampled) and checks that the per-timestep
//  posterior-mode state path recovers z*.
//
//  Honest checks:
//    (1) FFBS posterior-mode path matches z* on a high fraction of timesteps,
//        AND beats the naive 50% coin-flip baseline by a wide margin.
//    (2) The per-timestep posterior smoothing marginals P(z_t = z*_t) average
//        well above 0.5 (the sampler concentrates on the true states).
//==============================================================================
#include <cstdio>
int main() {
    // ---- Known truth -------------------------------------------------------
    const std::size_t T = 400;
    // Sticky transition matrix (row-major 2x2): stay w.p. 0.92, switch 0.08.
    const arma::vec A   = { 0.92, 0.08,
                            0.08, 0.92 };
    const arma::vec pi0 = { 0.5, 0.5 };
    const arma::vec mu  = { -2.0, 2.0 };   // well-separated relative to sigma
    const double    sigma = 1.0;

    // ---- Simulate latent path z* and emissions y --------------------------
    std::mt19937_64 sim_rng(2026);
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    std::normal_distribution<double>       gz(0.0, 1.0);

    arma::vec z_true(T);
    arma::vec y(T);

    // initial state from pi0
    std::size_t z_prev = (unif(sim_rng) < pi0[0]) ? 0u : 1u;
    z_true[0] = static_cast<double>(z_prev);
    y[0]      = mu[z_prev] + sigma * gz(sim_rng);
    for (std::size_t t = 1; t < T; ++t) {
        // transition: A row z_prev
        const double stay = A[z_prev * 2 + z_prev];
        std::size_t z_cur;
        if (unif(sim_rng) < stay) z_cur = z_prev;
        else                      z_cur = 1u - z_prev;
        z_true[t] = static_cast<double>(z_cur);
        y[t]      = mu[z_cur] + sigma * gz(sim_rng);
        z_prev    = z_cur;
    }

    // ---- Fit: only z is sampled (A, pi, mu, sigma fixed at truth) ---------
    HMMGaussian2State model(y, A, pi0, mu, sigma, /*rng_seed=*/7,
                            /*keep_history=*/false);
    model.step(200);   // warmup sweeps (FFBS is exact; just burn label noise)

    // Accumulate per-timestep posterior smoothing marginals P(z_t = 1).
    const int M = 2000;
    arma::vec p1(T, arma::fill::zeros);
    for (int s = 0; s < M; ++s) {
        model.step(1);
        const arma::vec z = model.get_current().at("z");   // copy (temp map)
        for (std::size_t t = 0; t < T; ++t)
            if (static_cast<int>(z[t]) == 1) p1[t] += 1.0;
    }
    p1 /= static_cast<double>(M);

    // ---- Posterior-mode path + recovery metrics ---------------------------
    std::size_t n_correct = 0;
    double      sum_p_at_truth = 0.0;
    for (std::size_t t = 0; t < T; ++t) {
        const int z_hat   = (p1[t] >= 0.5) ? 1 : 0;
        const int z_star  = static_cast<int>(z_true[t]);
        if (z_hat == z_star) ++n_correct;
        sum_p_at_truth += (z_star == 1) ? p1[t] : (1.0 - p1[t]);
    }
    const double acc        = static_cast<double>(n_correct) / static_cast<double>(T);
    const double mean_p_true = sum_p_at_truth / static_cast<double>(T);
    const double naive_acc   = 0.5;   // coin-flip baseline

    std::printf("HMMGaussian2State demo: T=%zu  sweeps=%d\n", T, M);
    std::printf("  posterior-mode path accuracy = %.3f  (naive baseline %.2f)\n",
                acc, naive_acc);
    std::printf("  mean P(z_t = z*_t)           = %.3f\n", mean_p_true);

    // Well-separated means (|mu|=2, sigma=1) + sticky chain => FFBS should
    // recover the path on the large majority of timesteps and dominate the
    // coin-flip baseline; marginals should concentrate above 0.5 at truth.
    const bool ok = (acc > 0.85) && (acc > naive_acc + 0.30) &&
                    (mean_p_true > 0.80);
    std::printf("%s\n",
                ok ? "[demo PASS] FFBS recovers the latent state path"
                   : "[demo FAIL] FFBS did not recover the latent path");
    return ok ? 0 : 1;
}

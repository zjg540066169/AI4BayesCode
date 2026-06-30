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

// @example:R
//   library(AI4BayesCode)
//   ai4bayescode_example("HMMGaussian2State")
//   set.seed(2026)
//   T   <- 400                                  # length of the state sequence
//   A   <- c(0.92, 0.08, 0.08, 0.92)            # sticky 2x2 transition (row-major)
//   pi0 <- c(0.5, 0.5)                          # initial distribution over states
//   mu  <- c(-2, 2)                             # well-separated emission means
//   sigma <- 1                                  # common emission sd (|mu|=2 >> sigma)
//   # simulate the latent path z* and Gaussian emissions y
//   z <- integer(T); z[1] <- sample.int(2, 1, prob = pi0) - 1L
//   for (t in 2:T) z[t] <- if (runif(1) < A[z[t-1]*2 + z[t-1] + 1]) z[t-1] else 1L - z[t-1]
//   y <- mu[z + 1L] + sigma * rnorm(T)
//   m <- new(HMMGaussian2State,
//            y,        # length-T observations
//            A,        # length-4 row-major transition matrix
//            pi0,      # length-2 initial distribution
//            mu,       # length-2 emission means (fixed)
//            sigma,    # emission sd (fixed)
//            7L,       # RNG seed
//            TRUE)     # keep_history
//   m$step(2500); str(m$get_current())          # only z (latent path) is sampled
// @example:python
//   import numpy as np, AI4BayesCode
//   rng = np.random.default_rng(2026)
//   T   = 400                                     # length of the state sequence
//   A   = np.array([0.92, 0.08, 0.08, 0.92])      # sticky 2x2 transition (row-major)
//   pi0 = np.array([0.5, 0.5])                     # initial distribution over states
//   mu  = np.array([-2.0, 2.0])                    # well-separated emission means
//   sigma = 1.0                                    # common emission sd (|mu|=2 >> sigma)
//   z = np.empty(T, dtype=int)
//   z[0] = 0 if rng.uniform() < pi0[0] else 1
//   for t in range(1, T):
//       z[t] = z[t-1] if rng.uniform() < A[z[t-1]*2 + z[t-1]] else 1 - z[t-1]
//   y = mu[z] + sigma * rng.standard_normal(T)     # Gaussian emissions
//   Mod = AI4BayesCode.source("HMMGaussian2State.cpp")
//   m = Mod.HMMGaussian2State(y, A, pi0, mu, sigma, 7, True)  # (y, A, pi, mu, sigma, seed, keep_history)
//   m.step(2500); print(m.get_current())           # dict: z (latent path) only
// @example:end

// [[Rcpp::depends(RcppArmadillo)]]

#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif

#ifdef AI4BAYESCODE_RCPP_MODULE
#  include <RcppArmadillo.h>
#else
#  include <armadillo>
#endif

#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/backend_neutral.hpp"
#include "AI4BayesCode/shared_data.hpp"
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/rcpp_wrap.hpp"
#include "AI4BayesCode/hmm_block.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>

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
        if (y.n_elem < 2) ai4b::stop("y length must be >= 2");
        if (A_flat_row_major.n_elem != 4)
            ai4b::stop("A must be length 4 (row-major 2x2)");
        if (pi_init.n_elem != 2) ai4b::stop("pi must be length 2");
        if (mu.n_elem != 2)      ai4b::stop("mu must be length 2");
        if (!(sigma > 0.0))      ai4b::stop("sigma must be > 0");
        // Validate row sums = 1 on A.
        if (std::abs((A_flat_row_major[0] + A_flat_row_major[1]) - 1.0) > 1e-8 ||
            std::abs((A_flat_row_major[2] + A_flat_row_major[3]) - 1.0) > 1e-8) {
            ai4b::stop("A rows must sum to 1");
        }
        if (std::abs((pi_init[0] + pi_init[1]) - 1.0) > 1e-8) {
            ai4b::stop("pi must sum to 1");
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

    // ---- Canonical 6-method R interface ----

    void step(int n_steps) {
        if (n_steps < 0) ai4b::stop("n_steps must be >= 0");
        for (int i = 0; i < n_steps; ++i) impl_->step(rng_);
    }

    // Backend-neutral state_map: only z (the latent path) is sampled.
    // z is stored as a flat length-T arma::vec (state labels 0/1).
    AI4BayesCode::state_map get_current() const {
        AI4BayesCode::state_map out;
        out["z"] = impl_->data().get("z");      // length T
        return out;
    }

    // Backend-neutral state_map input. Accepts "z" (length T latent path) and
    // optionally "y" (length T observations). Each value is an arma::vec
    // (frontend already converted R list / Python dict).
    void set_current(const AI4BayesCode::state_map& params) {
        auto* z_blk = dynamic_cast<hmm_block*>(&impl_->child(0));
        auto it_z = params.find("z");
        if (it_z != params.end()) {
            const arma::vec& znew = it_z->second;
            if (znew.n_elem != T_)
                ai4b::stop("set_current: z length must equal T");
            z_blk->set_current(znew);
            impl_->data().set("z", znew);
        }
        auto it_y = params.find("y");
        if (it_y != params.end()) {
            const arma::vec& y_new = it_y->second;
            if (y_new.n_elem != T_) ai4b::stop("y length must equal T");
            impl_->data().set("y", y_new);
        }
    }

    // Backend-neutral history_map output. HMMGaussian2State has no covariate
    // inputs, so predict_at() takes an EMPTY map and returns posterior
    // predictive y_rep. Every key maps to an arma::mat: keep_history = FALSE
    // returns 1-row matrices (single predict at the current draw);
    // keep_history = TRUE returns n_draws-row matrices (posterior predictive
    // over every recorded z draw). mu, sigma are FIXED at construction, so
    // only z is replayed.
    AI4BayesCode::history_map predict_at(
            const AI4BayesCode::state_map& new_data) const {
        if (!new_data.empty())
            ai4b::stop("HMMGaussian2State: predict_at takes an empty map/list "
                       "(no covariate inputs).");

        AI4BayesCode::history_map out;

        if (!keep_history_) {
            block_context replaced;
            block_context result = impl_->predict_at(replaced, predict_rng_);
            for (const auto& kv : result) {
                arma::mat m(1, kv.second.n_elem);
                for (std::size_t j = 0; j < kv.second.n_elem; ++j)
                    m(0, j) = kv.second[j];
                out.emplace(kv.first, std::move(m));
            }
            return out;
        }

        // History mode: loop over all posterior draws of z.
        // mu, sigma are fixed in the constructor (not sampled), so we only
        // need to replay z. z IS the block name of hmm_block.
        AI4BayesCode::history_map hist = impl_->get_history();
        auto it_z = hist.find("z");
        if (it_z == hist.end()) {
            ai4b::stop("HMMGaussian2State::predict_at: keep_history_ requires "
                       "z history but get_history() lacks it.");
        }
        const arma::mat& z_hist  = it_z->second;       // n_draws x T
        const std::size_t n_draws = z_hist.n_rows;

        std::unordered_map<std::string, std::vector<arma::vec>> collected;
        for (std::size_t d = 0; d < n_draws; ++d) {
            block_context replaced;
            replaced["z"] = z_hist.row(d).t();
            block_context result = impl_->predict_at(replaced, predict_rng_);
            for (const auto& kv : result)
                collected[kv.first].push_back(kv.second);
        }

        for (const auto& kv : collected) {
            if (kv.second.empty()) continue;
            const std::size_t dim = kv.second[0].n_elem;
            arma::mat mat(n_draws, dim);
            for (std::size_t d = 0; d < n_draws; ++d)
                for (std::size_t i = 0; i < dim; ++i)
                    mat(d, i) = kv.second[d][i];
            out.emplace(kv.first, std::move(mat));
        }
        return out;
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

#ifdef AI4BAYESCODE_RCPP_MODULE
RCPP_MODULE(HMMGaussian2State_module) {
    Rcpp::class_<HMMGaussian2State>("HMMGaussian2State")
        .constructor<arma::vec, arma::vec, arma::vec, arma::vec, double,
                     int>(
            "Legacy constructor; keep_history defaults to FALSE.")
        .constructor<arma::vec, arma::vec, arma::vec, arma::vec, double,
                     int, bool>(
            "Construct with y (length T observations), A (length 4 "
            "transition row-major), pi (length 2 initial), mu (length 2 "
            "emission means), sigma (common emission sd), seed, "
            "keep_history. Runs forward-filter backward-sample on z "
            "via hmm_block (Check #15 parity test at "
            "tests_autodiff/test_hmm_block.cpp).")
        .method("step",        &HMMGaussian2State::step)
        .method("get_current", &HMMGaussian2State::get_current)
        .method("set_current", &HMMGaussian2State::set_current,
                "Overwrite z and/or y from a named list.")
        .method("predict_at",  &HMMGaussian2State::predict_at,
                "Posterior predictive y_rep given current z sequence.")
        .method("get_dag",     &HMMGaussian2State::get_dag)
        .method("get_history", &HMMGaussian2State::get_history);
}
#endif

#ifdef AI4BAYESCODE_PYBIND_MODULE
#include "AI4BayesCode/pybind_casters.hpp"
PYBIND11_MODULE(HMMGaussian2State, m) {
    AI4BayesCode::register_ai4bayescode_types(m);
    pybind11::class_<HMMGaussian2State>(m, "HMMGaussian2State")
        .def(pybind11::init<arma::vec, arma::vec, arma::vec, arma::vec,
                            double, int, bool>(),
             pybind11::arg("y"),
             pybind11::arg("A"),
             pybind11::arg("pi"),
             pybind11::arg("mu"),
             pybind11::arg("sigma"),
             pybind11::arg("rng_seed") = 1,
             pybind11::arg("keep_history") = false)
        .def("step",        &HMMGaussian2State::step, pybind11::arg("n_steps"))
        .def("get_current", &HMMGaussian2State::get_current)
        .def("set_current", &HMMGaussian2State::set_current,
             pybind11::arg("params"))
        .def("predict_at",  &HMMGaussian2State::predict_at,
             pybind11::arg("new_data"))
        .def("get_dag",     &HMMGaussian2State::get_dag)
        .def("get_history", &HMMGaussian2State::get_history);
}
#endif

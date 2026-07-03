// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
//  BetaBernoulli.cpp
//
//  Test example for sampling a Beta-distributed parameter p in (0, 1)
//  using NUTS with constraints::interval(0, 1).
//
//  This tests a common failure mode of AI-generated code: parameters on
//  [0, 1] require interval constraints, not real or positive transforms.
//
//  Model
//  -----
//      y_i | p  ~ Bernoulli(p),   i = 1..N
//      p        ~ Beta(a, b)
//
//  Analytic posterior: p | y ~ Beta(a + sum(y), b + N - sum(y))
// ============================================================================

// @example:R
//   library(AI4BayesCode)
//   ai4bayescode_example("BetaBernoulli")
//   set.seed(2026)
//   N <- 500; p_true <- 0.30            # N Bernoulli trials, true success prob
//   y <- as.numeric(runif(N) < p_true)  # 0/1 data vector
//   # new(BetaBernoulli, y, a, b, seed, keep_history):
//   #   y = data, a = 2 / b = 2 Beta prior shapes, seed = 7, keep_history = TRUE
//   # ---- Recommended: parallel chains + convergence diagnosis ----
//   run <- ai4bayescode_run_chains(
//       function(seed) new(BetaBernoulli, y, 2, 2, seed, TRUE),
//       n_chains = 4, n_burn = 1000, n_keep = 2000)
//   ai4bayescode_diagnose(run$histories[[1]])      # summary + R-hat/ESS + plots
//   # ---- Advanced: stateful single-chain control ----
//   m <- new(BetaBernoulli, y, 2, 2, 7L, TRUE)
//   m$step(2500); str(m$get_current())  # posterior mean p ~= (a+sum_y)/(a+b+N)
// @example:python
//   import numpy as np, AI4BayesCode
//   rng = np.random.default_rng(2026)
//   N, p_true = 500, 0.30
//   y = (rng.random(N) < p_true).astype(float)        # 0/1 data vector
//   Mod = AI4BayesCode.example("BetaBernoulli")
//   # BetaBernoulli(y, a, b, rng_seed, keep_history): a=2/b=2 Beta prior, seed=7
//   # ---- Recommended: parallel chains + diagnosis ----
//   chains = AI4BayesCode.run_chains(
//       lambda seed: Mod.BetaBernoulli(y, 2.0, 2.0, seed, True),
//       seeds=[101, 202, 303, 404], n_burn=1000, n_keep=2000, n_jobs=1)
//   AI4BayesCode.diagnose(chains[0]["hist"])   # summary + diagnostics
//   # ---- Advanced: stateful single-chain control ----
//   m = Mod.BetaBernoulli(y, 2.0, 2.0, 7, True)
//   m.step(2500); print(m.get_current())
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
#include "AI4BayesCode/nuts_block.hpp"
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/constraints.hpp"
#include "AI4BayesCode/rcpp_wrap.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>

using AI4BayesCode::block_context;
using AI4BayesCode::composite_block;
using AI4BayesCode::nuts_block;
using AI4BayesCode::nuts_block_config;

namespace constraints = AI4BayesCode::constraints;

namespace {

// --------------------------------------------------------------------------
//  p (scalar, in (0, 1))
//
//  Natural-scale log density:
//      log p(p | y, a, b) = (a + sum_y - 1)*log(p) + (b + N - sum_y - 1)*log(1-p)
//
//  Gradient on natural (p) scale:
//      d/dp = (a + sum_y - 1)/p - (b + N - sum_y - 1)/(1-p)
//
//  The interval constraint Jacobian (logit transform) is added by
//  constraints::interval::wrap. Do NOT include it here.
// --------------------------------------------------------------------------
double p_log_density(const arma::vec& p_nat,
                     const block_context& ctx,
                     arma::vec* grad_nat) {
    const double p     = p_nat[0];
    const double sum_y = ctx.at("sum_y")[0];
    const double N     = ctx.at("N")[0];
    const double a     = ctx.at("a")[0];
    const double b     = ctx.at("b")[0];

    // Boundary check
    if (p <= 0.0 || p >= 1.0) {
        return -std::numeric_limits<double>::infinity();
    }

    const double alpha_post = a + sum_y - 1.0;
    const double beta_post  = b + N - sum_y - 1.0;

    const double lp = alpha_post * std::log(p)
                    + beta_post  * std::log(1.0 - p);

    if (grad_nat) {
        grad_nat->set_size(1);
        (*grad_nat)[0] = alpha_post / p - beta_post / (1.0 - p);
    }
    return lp;
}

} // anonymous namespace

class BetaBernoulli {
public:
    BetaBernoulli(const arma::vec& y, double a, double b, int rng_seed,
                  bool keep_history = false)
        : rng_(rng_seed == 0
                   ? std::mt19937_64{std::random_device{}()}
                   : std::mt19937_64{static_cast<std::uint64_t>(rng_seed)}),
          predict_rng_(rng_seed == 0
                   ? std::mt19937_64{std::random_device{}()}
                   : std::mt19937_64{static_cast<std::uint64_t>(rng_seed)
                                     ^ 0x9E3779B97F4A7C15ULL}),
          readapt_rng_(rng_seed == 0
                   ? std::mt19937_64{std::random_device{}()}
                   : std::mt19937_64{static_cast<std::uint64_t>(rng_seed)
                                     ^ 0xBF58476D1CE4E5B9ULL}),
          impl_(std::make_unique<composite_block>("BetaBernoulli")),
          keep_history_(keep_history)
    {
        const double N     = static_cast<double>(y.n_elem);
        const double sum_y = arma::sum(y);

        // Prior mean as initial p
        double p_init = a / (a + b);
        // Clamp away from boundaries
        p_init = std::max(0.01, std::min(0.99, p_init));

        impl_->data().set("sum_y", arma::vec{sum_y});
        impl_->data().set("N",     arma::vec{N});
        impl_->data().set("a",     arma::vec{a});
        impl_->data().set("b",     arma::vec{b});
        impl_->data().set("p",     arma::vec{p_init});

        impl_->data().declare_dependencies("p", {"sum_y", "N", "a", "b"});

        // ---- Predict DAG: p -> y_rep (iid Bernoulli posterior predictive)
        impl_->data().declare_predict_edges("p", {"y_rep"});

        // ---- Generative-DAG context (VIZ-ONLY; predict_at BFS never
        //      reads context_edges_). p ~ Beta(a, b): the Beta
        //      hyperparameters are p's prior parents, drawn faded by
        //      ai4bayescode_plot_dag so the full generative story is visible.
        impl_->data().declare_context_edges("a", {"p"});
        impl_->data().declare_context_edges("b", {"p"});

        impl_->data().set("y_rep", arma::vec(y.n_elem, arma::fill::zeros));

        const std::size_t N_obs = y.n_elem;
        impl_->data().register_stochastic_refresher(
            "y_rep",
            [N_obs](const AI4BayesCode::shared_data_t& d,
                    std::mt19937_64& rng) {
                const double p = d.get("p")[0];
                std::bernoulli_distribution bern(p);
                arma::vec y_rep(N_obs);
                for (std::size_t i = 0; i < N_obs; ++i) {
                    y_rep[i] = bern(rng) ? 1.0 : 0.0;
                }
                return y_rep;
            });

        nuts_block_config cfg;
        cfg.name = "p";

        // CRITICAL: use interval(0, 1) constraint, not positive or real.
        // p lives in (0, 1), so the unconstraining transform is logit.
        cfg.initial_unc = constraints::interval::unconstrain(
            arma::vec{p_init}, 0.0, 1.0);
        cfg.constrain = [](const arma::vec& unc) {
            return constraints::interval::constrain(unc, 0.0, 1.0);
        };
        cfg.unconstrain = [](const arma::vec& nat) {
            return constraints::interval::unconstrain(nat, 0.0, 1.0);
        };
        cfg.log_density_grad =
            [](const arma::vec& p_unc, const block_context& ctx,
               arma::vec* grad) {
                return constraints::interval::wrap(
                    p_unc, grad, 0.0, 1.0,
                    [&](const arma::vec& p_nat, arma::vec* grad_nat) {
                        return p_log_density(p_nat, ctx, grad_nat);
                    });
            };

        impl_->add_child(std::make_unique<nuts_block>(std::move(cfg)));

        // ---- Enable history recording if requested ----------------------
        if (keep_history_) {
            impl_->set_keep_history(true);
        }
    }

    void step() { step(1); }              // no-arg convenience: one sweep
    void step(int n_steps) {
        for (int i = 0; i < n_steps; ++i) impl_->step(rng_);
    }

    AI4BayesCode::state_map get_current() const {
        AI4BayesCode::state_map out;
        out["p"] = impl_->data().get("p");
        return out;
    }

    void set_current(const AI4BayesCode::state_map& params) {
        auto it = params.find("p");
        if (it != params.end()) {
            const double pv = it->second[0];
            if (pv <= 0 || pv >= 1)
                throw std::runtime_error("p must be in (0, 1)");
            dynamic_cast<nuts_block&>(impl_->child(0)).set_current(
                arma::vec{pv});
            impl_->data().set("p", arma::vec{pv});
        }
    }

    // predict_at: posterior predictive y_rep (length N iid Bernoulli(p)).
    // Empty input map. Non-history mode -> 1xN matrix; history mode ->
    // n_draws x N matrix.
    AI4BayesCode::history_map predict_at(const AI4BayesCode::state_map& new_data) const {
        if (!new_data.empty())
            throw std::runtime_error("BetaBernoulli has no covariate inputs.");

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

        AI4BayesCode::history_map hist = impl_->get_history();
        const arma::mat& p_hist = hist.at("p");
        const std::size_t n_draws = p_hist.n_rows;

        std::unordered_map<std::string, std::vector<arma::vec>> collected;
        for (std::size_t d = 0; d < n_draws; ++d) {
            block_context replaced;
            replaced["p"] = arma::vec{p_hist(d, 0)};
            block_context result = impl_->predict_at(replaced, predict_rng_);
            for (const auto& kv : result) {
                collected[kv.first].push_back(kv.second);
            }
        }

        for (const auto& kv : collected) {
            if (kv.second.empty()) continue;
            const std::size_t dim = kv.second[0].n_elem;
            arma::mat m(n_draws, dim);
            for (std::size_t i = 0; i < n_draws; ++i)
                for (std::size_t j = 0; j < dim; ++j)
                    m(i, j) = kv.second[i][j];
            out.emplace(kv.first, std::move(m));
        }
        return out;
    }

    // Full model DAG (gibbs_reads + gibbs_invalidates + predict_edges +
    // data_inputs). See composite_block::get_dag().
    AI4BayesCode::dag_info get_dag() const { return impl_->get_dag(); }

    // ---- History access --------------------------------------------------
    AI4BayesCode::history_map get_history() const { return impl_->get_history(); }

    /// 7th R-level method: re-tune NUTS metric (mass matrix + step size +
    /// dual averaging) without advancing chain state. Available because
    /// the composite contains NUTS-family children. See system_design.md
    /// §13 NUTS-family + validator.md §24.
    void readapt_NUTS(int n, bool reset = false, int max_tree_depth = -1) {
        if (n < 0) {
            ai4b::stop("readapt_NUTS: n must be non-negative");
        }
        impl_->readapt_NUTS(static_cast<std::size_t>(n),
                            reset, readapt_rng_, max_tree_depth < 0 ? std::size_t(0) : static_cast<std::size_t>(max_tree_depth));
    }


private:
    std::mt19937_64                  rng_;
    mutable std::mt19937_64          predict_rng_;
    mutable std::mt19937_64          readapt_rng_; // readapt_NUTS() advances it (7th method)
    std::unique_ptr<composite_block> impl_;
    bool                             keep_history_ = false;
};

#ifdef AI4BAYESCODE_RCPP_MODULE
RCPP_MODULE(BetaBernoulli_module) {
    Rcpp::class_<BetaBernoulli>("BetaBernoulli")
        .constructor<arma::vec, double, double, int>(
            "Legacy constructor; keep_history defaults to FALSE.")
        .constructor<arma::vec, double, double, int, bool>(
            "Construct with: y (0/1 vector), Beta prior a, b, seed, "
            "keep_history (default FALSE).")
        .method("step", (void (BetaBernoulli::*)())    &BetaBernoulli::step, "Run one sweep.")
        .method("step", (void (BetaBernoulli::*)(int)) &BetaBernoulli::step, "Run n sweeps.")
        .method("get_current", &BetaBernoulli::get_current)
        .method("set_current", &BetaBernoulli::set_current)
        .method("predict_at",  &BetaBernoulli::predict_at)
        .method("get_dag",     &BetaBernoulli::get_dag)
        .method("get_history", &BetaBernoulli::get_history)
        .method("readapt_NUTS", &BetaBernoulli::readapt_NUTS);
}
#endif

#ifdef AI4BAYESCODE_PYBIND_MODULE
#include "AI4BayesCode/pybind_casters.hpp"
PYBIND11_MODULE(BetaBernoulli, m) {
    AI4BayesCode::register_ai4bayescode_types(m);
    pybind11::class_<BetaBernoulli>(m, "BetaBernoulli")
        .def(pybind11::init<arma::vec, double, double, int, bool>(),
             pybind11::arg("y"),
             pybind11::arg("a") = 1.0, pybind11::arg("b") = 1.0,
             pybind11::arg("rng_seed") = 1,
             pybind11::arg("keep_history") = false,
             "Bayesian Beta-Bernoulli conjugate. p ~ Beta(a,b); "
             "y ~ Bernoulli(p). Uses beta_gibbs_block (exact).")
        .def("step", (void (BetaBernoulli::*)())    &BetaBernoulli::step, "Run one sweep.")
        .def("step", (void (BetaBernoulli::*)(int)) &BetaBernoulli::step, pybind11::arg("n_steps"))
        .def("get_current", &BetaBernoulli::get_current)
        .def("set_current", &BetaBernoulli::set_current, pybind11::arg("params"))
        .def("predict_at",  &BetaBernoulli::predict_at, pybind11::arg("new_data"))
        .def("get_dag",     &BetaBernoulli::get_dag)
        .def("get_history", &BetaBernoulli::get_history)
        .def("readapt_NUTS", &BetaBernoulli::readapt_NUTS,
             pybind11::arg("n"), pybind11::arg("reset") = false, pybind11::arg("max_tree_depth") = -1);
}
#endif

#if !defined(AI4BAYESCODE_RCPP_MODULE) && !defined(AI4BAYESCODE_PYBIND_MODULE)
#include <cstdio>
// ============================================================================
//  Standalone demo: simulate Bernoulli(p_true) data, fit p with NUTS under the
//  interval(0, 1) constraint, and confirm the sampled posterior mean recovers
//  both the analytic Beta-posterior mean and the data-generating p_true.
//
//  Analytic check: with prior Beta(a, b),
//      p | y ~ Beta(a + sum_y, b + N - sum_y),  posterior mean = (a+sum_y)/(a+b+N).
//  The NUTS posterior mean must match this closed form (the real correctness
//  test) and also land near p_true.
// ============================================================================
int main() {
    const std::size_t N      = 400;
    const double      p_true  = 0.30;
    const double      a       = 2.0;   // Beta prior shape
    const double      b       = 2.0;   // Beta prior shape

    // ---- Simulate Bernoulli(p_true) data ------------------------------------
    std::mt19937_64 sim_rng(123);
    std::bernoulli_distribution gen(p_true);
    arma::vec y(N);
    double sum_y = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        y[i] = gen(sim_rng) ? 1.0 : 0.0;
        sum_y += y[i];
    }

    // ---- Analytic posterior (the ground truth for the sampler) --------------
    const double post_a    = a + sum_y;
    const double post_b    = b + static_cast<double>(N) - sum_y;
    const double post_mean = post_a / (post_a + post_b);

    // ---- Fit p with constrained NUTS ----------------------------------------
    BetaBernoulli model(y, a, b, /*rng_seed=*/7, /*keep_history=*/false);
    model.step(800);   // warmup + adaptation

    double p_bar = 0.0;
    const int M = 4000;
    for (int s = 0; s < M; ++s) {
        model.step(1);
        p_bar += model.get_current().at("p")[0];
    }
    p_bar /= static_cast<double>(M);

    std::printf("BetaBernoulli demo: N=%zu  sum_y=%.0f  p_true=%.3f\n",
                N, sum_y, p_true);
    std::printf("  NUTS posterior mean p_hat = %.4f\n", p_bar);
    std::printf("  analytic posterior mean   = %.4f\n", post_mean);
    std::printf("  abs err vs analytic       = %.4f\n",
                std::abs(p_bar - post_mean));
    std::printf("  abs err vs p_true         = %.4f\n",
                std::abs(p_bar - p_true));

    // The sampler is correct iff it reproduces the closed-form posterior mean
    // (tight tolerance); recovering p_true is the looser sanity check.
    const bool ok = std::abs(p_bar - post_mean) < 0.01 &&
                    std::abs(p_bar - p_true)    < 0.05;
    std::printf("%s\n",
                ok ? "[demo PASS] constrained NUTS matches analytic Beta posterior"
                   : "[demo FAIL] posterior mean off");
    return ok ? 0 : 1;
}
#endif

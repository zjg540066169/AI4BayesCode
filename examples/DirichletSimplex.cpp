// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v3.0 or later
// (GPL-3.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
//  DirichletSimplex_v2.cpp
//
//  JOINT-NUTS rewrite of DirichletSimplex.cpp. Same model, same priors,
//  same posterior — but theta is sampled by ONE joint_nuts_block with a
//  single SIMPLEX sub-param instead of a single nuts_block with
//  constraints::simplex::wrap. This is the trivial conversion case: only one
//  continuous parameter block, so the joint block holds exactly one sub-param.
//
//  Model (identical to DirichletSimplex.cpp)
//  -----------------------------------------
//      theta       ~ Dirichlet(alpha)                  (on the K-simplex)
//      y_i | theta ~ Categorical(theta),   i = 1..N
//
//  (Equivalently, y_counts | theta ~ Multinomial(N, theta).)
//
//  Conjugate posterior:
//      theta | y_counts ~ Dirichlet(alpha + y_counts)
//
//  JOINT natural-scale log-density over theta (length K, simplex):
//
//    log p(theta | y_counts, alpha) = sum_k (alpha_k + y_k - 1) * log(theta_k)
//    grad_k = (alpha_k + y_k - 1) / theta_k
//
//  joint_nuts_block adds the SIMPLEX stick-breaking Jacobian internally;
//  this function operates on the NATURAL simplex scale. Do NOT include
//  Jacobians here.
//
//  Cross-validated against DirichletSimplex.cpp: posteriors must match
//  (means, sd, R-hat). Both files can be loaded in one R session because
//  they use distinct class and module names.
// ============================================================================

// @example:R
//   library(AI4BayesCode)
//   ai4bayescode_example("DirichletSimplex")
//   set.seed(2026)
//   K <- 4L; N <- 500L                          # 4 categories, 500 obs
//   theta_true <- c(0.50, 0.25, 0.15, 0.10)     # known simplex point
//   cats  <- sample.int(K, N, replace = TRUE, prob = theta_true)
//   y     <- tabulate(cats, nbins = K)          # category counts (length K)
//   alpha <- rep(1, K)                          # flat Dirichlet prior
//   # ---- Recommended: parallel chains + convergence diagnosis ----
//   run <- ai4bayescode_run_chains(
//       function(seed) new(DirichletSimplex, y, alpha, seed, TRUE),
//       n_chains = 4, n_burn = 1000, n_keep = 2000)
//   ai4bayescode_diagnose(run$histories[[1]])      # summary + R-hat/ESS + plots
//   # ---- Advanced: stateful single-chain control ----
//   m <- new(DirichletSimplex, y, alpha, 7L, TRUE)  # y_counts, alpha, seed, keep_history
//   m$step(2500); str(m$get_current())
// @example:python
//   import numpy as np, AI4BayesCode
//   rng = np.random.default_rng(2026)
//   K, N = 4, 500
//   theta_true = np.array([0.50, 0.25, 0.15, 0.10])
//   cats = rng.choice(K, size=N, p=theta_true)
//   y = np.bincount(cats, minlength=K).astype(float)   # category counts
//   alpha = np.ones(K)                                  # flat Dirichlet prior
//   Mod = AI4BayesCode.example("DirichletSimplex")
//   # ---- Recommended: parallel chains + diagnosis ----
//   chains = AI4BayesCode.run_chains(
//       lambda seed: Mod.DirichletSimplex(y, alpha, seed, True),
//       seeds=[101, 202, 303, 404], n_burn=1000, n_keep=2000, n_jobs=1)
//   AI4BayesCode.diagnose(chains[0]["hist"])   # summary + diagnostics
//   # ---- Advanced: stateful single-chain control ----
//   m = Mod.DirichletSimplex(y, alpha, 7, True)         # y_counts, alpha, seed, keep_history
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
#include "AI4BayesCode/joint_nuts_block.hpp"
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/constraints.hpp"
#include "AI4BayesCode/rcpp_wrap.hpp"
#include "AI4BayesCode/kernel_control_mixin.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>

using AI4BayesCode::block_context;
using AI4BayesCode::composite_block;
using AI4BayesCode::joint_nuts_block;
using AI4BayesCode::joint_nuts_block_config;
using AI4BayesCode::joint_nuts_sub_param;
using AI4BayesCode::joint_constraint;

namespace constraints = AI4BayesCode::constraints;

// ============================================================================
//  Joint natural-scale log-density for theta on the simplex.
//
//  theta_cat = [ theta (K entries, simplex) ].
//
//  log p(theta | y_counts, alpha) = sum_k (alpha_k + y_k - 1) * log(theta_k)
//  grad_k = (alpha_k + y_k - 1) / theta_k
//
//  joint_nuts_block adds the SIMPLEX stick-breaking Jacobian internally;
//  this function stays on the NATURAL scale. Do NOT include Jacobians here.
// ============================================================================

namespace {

double dirichlet_joint_log_density(const arma::vec& theta_cat,
                                   const block_context& ctx,
                                   arma::vec* grad_nat) {
    const arma::vec& y     = ctx.at("y");      // counts, length K
    const arma::vec& alpha = ctx.at("alpha");  // prior, length K
    const std::size_t K    = theta_cat.n_elem;

    double lp = 0.0;
    for (std::size_t k = 0; k < K; ++k) {
        const double tk = theta_cat[k];
        if (tk <= 0.0) {
            if (grad_nat) grad_nat->set_size(K);
            return -std::numeric_limits<double>::infinity();
        }
        lp += (alpha[k] + y[k] - 1.0) * std::log(tk);
    }

    if (grad_nat) {
        grad_nat->set_size(K);
        for (std::size_t k = 0; k < K; ++k) {
            (*grad_nat)[k] = (alpha[k] + y[k] - 1.0) / theta_cat[k];
        }
    }
    return lp;
}

} // anonymous namespace

// ============================================================================
//  User-facing class exposed to R.
// ============================================================================

class DirichletSimplex : public AI4BayesCode::kernel_control_mixin<DirichletSimplex> {
public:
    DirichletSimplex(const arma::vec& y_counts,
                       const arma::vec& alpha,
                       int    rng_seed,
                       bool   keep_history = false)
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
          impl_(std::make_unique<composite_block>("DirichletSimplex")),
          keep_history_(keep_history)
    {
        if (y_counts.n_elem != alpha.n_elem) {
            throw std::runtime_error("y_counts and alpha must have the same length K");
        }
        if (y_counts.n_elem < 2) {
            throw std::runtime_error("simplex requires K >= 2 categories");
        }

        const std::size_t K = y_counts.n_elem;

        // ---- Install fixed data and initial theta --------------------------
        impl_->data().set("y",     y_counts);
        impl_->data().set("alpha", alpha);

        // Start at the prior mean (= uniform when alpha is flat).
        const arma::vec theta_init = alpha / arma::sum(alpha);
        impl_->data().set("theta", theta_init);

        // ---- Declare the dependency DAG ------------------------------------
        // Joint block's dependencies are keyed under the JOINT BLOCK NAME
        // ("theta_joint"), not the sub-param name "theta". The composite
        // builds the block's context by looking up dependencies under the
        // block name. Data inputs are the true external shared_data entries:
        // y and alpha (theta comes from the block's own concatenated draw).
        impl_->data().declare_dependencies("theta_joint", {"y", "alpha"});

        // ---- Predict DAG: keyed by SUB-PARAM name "theta" (not block name).
        impl_->data().declare_predict_edges("theta", {"y_rep"});

        // ---- Generative-DAG context (VIZ-ONLY; predict_at BFS never reads).
        // alpha -> theta represents theta's Dirichlet prior parent.
        impl_->data().declare_context_edges("alpha", {"theta"});

        impl_->data().set("y_rep", arma::vec(K, arma::fill::zeros));
        const int N_total = static_cast<int>(arma::sum(y_counts));
        impl_->data().register_stochastic_refresher(
            "y_rep",
            [N_total](const AI4BayesCode::shared_data_t& d,
                      std::mt19937_64& rng) {
                const arma::vec& th = d.get("theta");
                const std::size_t Kk = th.n_elem;
                std::discrete_distribution<int> cat(th.begin(), th.end());
                arma::vec counts(Kk, arma::fill::zeros);
                for (int n = 0; n < N_total; ++n)
                    counts[static_cast<std::size_t>(cat(rng))] += 1.0;
                return counts;
            });

        // ---- ONE joint_nuts_block over theta (SIMPLEX) --------------------
        {
            joint_nuts_block_config cfg;
            cfg.name = "theta_joint";
            cfg.sub_params.push_back(
                joint_nuts_sub_param{ "theta", K, joint_constraint::SIMPLEX });
            // initial_cat is NATURAL-scale: theta_init (K entries, sums to 1).
            cfg.initial_cat = theta_init;
            cfg.log_density_grad = &dirichlet_joint_log_density;
            // Tight-constraint tuning: diagonal mass + extra warmup runway.
            cfg.use_diagonal_metric = true;
            cfg.n_warmup_first_call = 800;
            impl_->add_child(std::make_unique<joint_nuts_block>(std::move(cfg)));
        }

        if (keep_history_) impl_->set_keep_history(true);
    }

    void step() { step(1); }              // no-arg convenience: one sweep
    void step(int n_steps) {
        for (int i = 0; i < n_steps; ++i) impl_->step(rng_);
    }

    AI4BayesCode::state_map get_current() const {
        AI4BayesCode::state_map out;
        out["theta"] = impl_->data().get("theta");
        return out;
    }

    void set_current(const AI4BayesCode::state_map& params) {
        auto it = params.find("theta");
        if (it != params.end()) {
            const arma::vec& theta_new = it->second;
            if (theta_new.n_elem != impl_->data().get("theta").n_elem) {
                throw std::runtime_error("theta has wrong length");
            }
            auto& jblk = dynamic_cast<joint_nuts_block&>(impl_->child(0));
            jblk.set_current(theta_new);
            impl_->data().set("theta", theta_new);
        }
    }

    // Posterior-predictive: y_rep ~ Multinomial(sum(y_counts_train), theta).
    // Non-history -> 1 x K arma::mat; history -> n_draws x K arma::mat.
    AI4BayesCode::history_map predict_at(const AI4BayesCode::state_map& new_data) const {
        if (!new_data.empty()) {
            throw std::runtime_error(
                "DirichletSimplex has no covariate inputs. "
                "predict_at() takes an empty map/list.");
        }
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

        // History mode: theta is a joint sub-output keyed by sub-param name "theta".
        AI4BayesCode::history_map hist = impl_->get_history();
        const arma::mat& th_hist  = hist.at("theta");
        const std::size_t n_draws = th_hist.n_rows;
        const std::size_t K       = th_hist.n_cols;

        std::unordered_map<std::string, std::vector<arma::vec>> collected;
        for (std::size_t d = 0; d < n_draws; ++d) {
            arma::vec th(K);
            for (std::size_t j = 0; j < K; ++j) th[j] = th_hist(d, j);
            block_context replaced;
            replaced["theta"] = th;
            block_context result = impl_->predict_at(replaced, predict_rng_);
            for (const auto& kv : result)
                collected[kv.first].push_back(kv.second);
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

    // ---- History access ----------------------------------------------------
    AI4BayesCode::history_map get_history() const { return impl_->get_history(); }

    void readapt_NUTS(int n, bool reset = false, int max_tree_depth = -1) {
        if (n < 0) ai4b::stop("readapt_NUTS: n must be non-negative");
        impl_->readapt_NUTS(static_cast<std::size_t>(n), reset, readapt_rng_, max_tree_depth < 0 ? std::size_t(0) : static_cast<std::size_t>(max_tree_depth));
    }

private:
    std::mt19937_64                  rng_;
    mutable std::mt19937_64          predict_rng_;
    mutable std::mt19937_64          readapt_rng_;
    std::unique_ptr<composite_block> impl_;
    bool                             keep_history_ = false;
};

// ============================================================================
//  Host-language modules (R + Python)
// ============================================================================

#ifdef AI4BAYESCODE_RCPP_MODULE
RCPP_MODULE(DirichletSimplex_module) {
    Rcpp::class_<DirichletSimplex>("DirichletSimplex")
        .constructor<arma::vec, arma::vec, int>(
            "Legacy constructor; keep_history defaults to FALSE.")
        .constructor<arma::vec, arma::vec, int, bool>(
            "Joint-NUTS DirichletSimplex: theta on K-simplex in one "
            "joint_nuts_block. Construct with category counts y_counts (length K), "
            "Dirichlet prior alpha (length K), RNG seed (0 = random), "
            "and keep_history (record every draw; default FALSE).")
        .method("step", (void (DirichletSimplex::*)())    &DirichletSimplex::step, "Run one sweep.")
        .method("step", (void (DirichletSimplex::*)(int)) &DirichletSimplex::step, "Run n sweeps.")
        .method("get_current",  &DirichletSimplex::get_current)
        .method("set_current",  &DirichletSimplex::set_current)
        .method("predict_at",   &DirichletSimplex::predict_at)
        .method("get_dag",      &DirichletSimplex::get_dag)
        .method("get_history",  &DirichletSimplex::get_history)
        .method("readapt_NUTS", &DirichletSimplex::readapt_NUTS)
        AI4BAYESCODE_BIND_KERNEL_CONTROL(DirichletSimplex);
}
#endif

#ifdef AI4BAYESCODE_PYBIND_MODULE
#include "AI4BayesCode/pybind_casters.hpp"
PYBIND11_MODULE(DirichletSimplex, m) {
    AI4BayesCode::register_ai4bayescode_types(m);
    pybind11::class_<DirichletSimplex>(m, "DirichletSimplex")
        .def(pybind11::init<arma::vec, arma::vec, int, bool>(),
             pybind11::arg("y_counts"), pybind11::arg("alpha"),
             pybind11::arg("rng_seed") = 1,
             pybind11::arg("keep_history") = false,
             "Joint-NUTS Dirichlet(alpha) prior on the simplex with multinomial "
             "likelihood. Uses joint_nuts_block on stick-breaking unconstraining.")
        .def("step", (void (DirichletSimplex::*)())    &DirichletSimplex::step, "Run one sweep.")
        .def("step", (void (DirichletSimplex::*)(int)) &DirichletSimplex::step,        pybind11::arg("n_steps"))
        .def("get_current",  &DirichletSimplex::get_current)
        .def("set_current",  &DirichletSimplex::set_current, pybind11::arg("params"))
        .def("predict_at",   &DirichletSimplex::predict_at,  pybind11::arg("new_data"))
        .def("get_dag",      &DirichletSimplex::get_dag)
        .def("get_history",  &DirichletSimplex::get_history)
        .def("readapt_NUTS", &DirichletSimplex::readapt_NUTS,
             pybind11::arg("n"), pybind11::arg("reset") = false, pybind11::arg("max_tree_depth") = -1)
        AI4BAYESCODE_PYBIND_KERNEL_CONTROL(DirichletSimplex);
}
#endif

// ============================================================================
//  Standalone demo (frontend-independent)
//
//  Simulate N categorical draws from a known theta_true on the K-simplex, fit
//  the joint_nuts_block, and check that the posterior mean of theta recovers
//  the EXACT conjugate Dirichlet posterior mean:
//
//      theta | y_counts ~ Dirichlet(alpha + y_counts)
//      E[theta_k | y]   = (alpha_k + y_k) / sum_j(alpha_j + y_j)
//
//  We compare the sampled posterior mean against this closed form (the exact
//  truth for this model) and, as a sanity floor, against theta_true.
// ============================================================================
#if !defined(AI4BAYESCODE_RCPP_MODULE) && !defined(AI4BAYESCODE_PYBIND_MODULE)
#include <cstdio>
int main() {
    const std::size_t K = 4;
    const int         N = 500;   // number of categorical observations

    // Known truth on the simplex.
    arma::vec theta_true = { 0.50, 0.25, 0.15, 0.10 };

    // Simulate category counts y ~ Multinomial(N, theta_true).
    std::mt19937_64 sim_rng(123);
    std::discrete_distribution<int> cat(theta_true.begin(), theta_true.end());
    arma::vec y_counts(K, arma::fill::zeros);
    for (int n = 0; n < N; ++n)
        y_counts[static_cast<std::size_t>(cat(sim_rng))] += 1.0;

    // Flat Dirichlet prior (alpha = 1).
    arma::vec alpha(K, arma::fill::ones);

    // Closed-form conjugate posterior mean = the exact target.
    arma::vec post_mean_exact = (alpha + y_counts) / arma::sum(alpha + y_counts);

    // Fit with the joint_nuts_block sampler.
    DirichletSimplex model(y_counts, alpha, /*rng_seed=*/7, /*keep_history=*/false);
    model.step(500);   // extra warmup beyond the block's own runway

    arma::vec theta_bar(K, arma::fill::zeros);
    const int M = 3000;
    for (int s = 0; s < M; ++s) {
        model.step(1);
        const auto cur = model.get_current();
        theta_bar += cur.at("theta");
    }
    theta_bar /= static_cast<double>(M);

    // Compare sampled posterior mean to the exact conjugate posterior mean.
    double max_err_post = 0.0, max_err_true = 0.0;
    for (std::size_t k = 0; k < K; ++k) {
        max_err_post = std::max(max_err_post,
                                std::abs(theta_bar[k] - post_mean_exact[k]));
        max_err_true = std::max(max_err_true,
                                std::abs(theta_bar[k] - theta_true[k]));
    }

    std::printf("DirichletSimplex demo (K=%zu, N=%d):\n", K, N);
    std::printf("  %-10s %-10s %-12s %-10s\n",
                "k", "truth", "post_exact", "sampled");
    for (std::size_t k = 0; k < K; ++k)
        std::printf("  %-10zu %-10.4f %-12.4f %-10.4f\n",
                    k, theta_true[k], post_mean_exact[k], theta_bar[k]);
    std::printf("  max|sampled - post_exact| = %.4f\n", max_err_post);
    std::printf("  max|sampled - truth|      = %.4f\n", max_err_true);

    // The sampler must match the EXACT conjugate posterior mean closely; the
    // looser truth check is a secondary sanity floor.
    const bool ok = (max_err_post < 0.02) && (max_err_true < 0.05);
    std::printf("%s\n",
                ok ? "[demo PASS] joint-NUTS recovers Dirichlet posterior mean"
                   : "[demo FAIL] posterior mean off");
    return ok ? 0 : 1;
}
#endif

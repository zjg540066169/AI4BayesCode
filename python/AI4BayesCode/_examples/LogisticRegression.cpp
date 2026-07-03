// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
//  LogisticRegression.cpp
//
//  Bayesian logistic regression via Polya-Gamma augmentation
//  (Polson-Scott-Windle 2013 JASA).
//
//  Model
//  -----
//      y_i  ~ Bernoulli(sigmoid(X_i' beta))     i = 1..N
//      beta ~ N(0, sigma_prior^2 I)             (weakly informative)
//
//  Why pg_logistic_block?
//  ---------------------
//  Logistic regression with NUTS on beta (via `nuts_block` + interval
//  constraints on sigmoid) is feasible but slow — the gradient
//  sigmoid'(X beta) X is expensive, and NUTS needs many leapfrogs per
//  step to navigate correlated beta posterior.
//
//  Hard scope: LINEAR LOGISTIC ONLY — NOT LOGISTIC BART
//  ----------------------------------------------------
//  This example (and the underlying `pg_logistic_block`) is for
//  parametric linear logistic regression: y_i ~ Bernoulli(sigmoid(
//  X_i' beta)) with a FIXED design matrix X. DO NOT attempt to plug a
//  BART mean function f(X_i) in place of X_i' beta — PG augmentation
//  has a non-identifiability issue with BART tree kernels (the
//  PG-augmented pseudo-response κ = y - 0.5 breaks BART's
//  Gaussian-observation assumption that anchors its tree location).
//  For logistic BART, use `genbart_block` with `logistic_lik`
//  (see `examples/GBartLogistic.cpp`).
//  See `include/AI4BayesCode/pg_logistic_block.hpp` header for details.
//
//  Polson-Scott-Windle 2013 showed that introducing Polya-Gamma auxiliary
//  variables omega_i ~ PG(1, X_i' beta) gives an EXACT Gibbs sampler:
//      beta | omega, y, X  ~  Normal (closed-form Gaussian conditional)
//      omega_i | beta, X   ~  PG(1, X_i' beta)
//
//  This is 10-100x faster than NUTS for moderate p (< 1000). Scaling is
//  dominated by O(p^3) from the beta-block Cholesky; for very large p,
//  further tricks (diagonal approximations, variational) are needed.
//
//  BLOCK DECOMPOSITION
//  -------------------
//      beta : pg_logistic_block  (single block, owns ω internally)
//
//  JUSTIFICATION (Check #16): Exception 1 from codegen.md §2b — PG
//  augmentation introduces a discrete-measure component (ω) that is
//  sampled via library-blessed pg_logistic_block (the block owns the
//  PG sampler and the Gaussian beta update internally). The user does
//  NOT hand-write any sampling; the block is fully library-provided.
//  Check #15 parity test at
//    tests_autodiff/test_pg_logistic_block.cpp
//  (verifies PG(1, z) mean matches analytical + beta recovery).
// ============================================================================

// @example:R
//   library(AI4BayesCode)
//   ai4bayescode_example("LogisticRegression")
//   set.seed(2026); N <- 800; p <- 3; beta_true <- c(-0.5, 1.2, -0.8)
//   X <- cbind(1, matrix(rnorm(N * (p - 1)), N, p - 1))   # col1 intercept, rest N(0,1)
//   prob <- 1 / (1 + exp(-(X %*% beta_true)))             # sigmoid(X beta)
//   y <- as.numeric(runif(N) < prob)                      # Bernoulli 0/1
//   # ---- Recommended: parallel chains + convergence diagnosis ----
//   run <- ai4bayescode_run_chains(
//       function(seed) new(LogisticRegression, X, y, 10.0, seed, TRUE),
//       n_chains = 4, n_burn = 1000, n_keep = 2000)
//   ai4bayescode_diagnose(run$histories[[1]])      # summary + R-hat/ESS + plots
//   # ---- Advanced: stateful single-chain control ----
//   m <- new(LogisticRegression, X, y, 10.0, 7L, TRUE)    # X, y, prior_sd, seed, keep_history
//   m$step(2500); str(m$get_current())
// @example:python
//   import numpy as np, AI4BayesCode
//   rng = np.random.default_rng(2026); N, p = 800, 3; beta_true = np.array([-0.5, 1.2, -0.8])
//   X = np.column_stack([np.ones(N), rng.standard_normal((N, p - 1))])  # intercept + N(0,1)
//   prob = 1.0 / (1.0 + np.exp(-(X @ beta_true)))                       # sigmoid(X beta)
//   y = (rng.random(N) < prob).astype(float)                           # Bernoulli 0/1
//   Mod = AI4BayesCode.example("LogisticRegression")
//   # ---- Recommended: parallel chains + diagnosis ----
//   chains = AI4BayesCode.run_chains(
//       lambda seed: Mod.LogisticRegression(X, y, 10.0, seed, True),
//       seeds=[101, 202, 303, 404], n_burn=1000, n_keep=2000, n_jobs=1)
//   AI4BayesCode.diagnose(chains[0]["hist"])   # summary + diagnostics
//   # ---- Advanced: stateful single-chain control ----
//   m = Mod.LogisticRegression(X, y, 10.0, 7, True); m.step(2500); print(m.get_current())
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
#include "AI4BayesCode/pg_logistic_block.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>

using AI4BayesCode::block_context;
using AI4BayesCode::composite_block;
using AI4BayesCode::pg_logistic_block;
using AI4BayesCode::pg_logistic_block_config;

class LogisticRegression {
public:
    LogisticRegression(const arma::mat& X,
                       const arma::vec& y,
                       double           prior_sd,
                       int              rng_seed,
                       bool             keep_history = false)
        : rng_(rng_seed == 0
                   ? std::mt19937_64{std::random_device{}()}
                   : std::mt19937_64{static_cast<std::uint64_t>(rng_seed)}),
          predict_rng_(rng_seed == 0
                   ? std::mt19937_64{std::random_device{}()}
                   : std::mt19937_64{static_cast<std::uint64_t>(rng_seed)
                                     ^ 0x9E3779B97F4A7C15ULL}),
          impl_(std::make_unique<composite_block>("LogisticRegression")),
          keep_history_(keep_history)
    {
        const std::size_t N = X.n_rows;
        const std::size_t p = X.n_cols;
        if (N == 0)       throw std::runtime_error("X must have at least 1 row");
        if (p == 0)       throw std::runtime_error("X must have at least 1 column");
        if (y.n_elem != N) throw std::runtime_error("y length must equal X row count");
        if (!(prior_sd > 0.0)) throw std::runtime_error("prior_sd must be > 0");
        // Validate y ∈ {0, 1}.
        for (std::size_t i = 0; i < N; ++i) {
            if (y[i] != 0.0 && y[i] != 1.0)
                throw std::runtime_error("y must be 0/1 (Bernoulli)");
        }
        N_ = N; p_ = p;

        impl_->data().set("y", y);
        impl_->data().set("X", arma::vectorise(X));

        // Initial beta = 0.
        impl_->data().set("beta", arma::vec(p, arma::fill::zeros));

        impl_->data().declare_dependencies(
            "beta", {"y", "X"});

        // ---- Full predict-DAG reconstruction (no collapsed intermediates).
        //   eta  = X * beta          (deterministic intermediate)
        //   prob = sigmoid(eta)      (deterministic intermediate)
        //   y_rep ~ Bernoulli(prob)  (stochastic; reads ONLY prob)
        // eta/prob are kept CURRENT during sampling via
        // declare_invalidates("beta", {eta, prob}) (order matters: eta
        // before prob). This is required because stateful
        // predict_at(list()) has an empty changed-set, so Pass-1 does NOT
        // recompute deterministic nodes — y_rep's Pass-2 must read an
        // already-current prob. Behaviour-preserving vs the old collapsed
        // y_rep (same lin = X*beta, same per-i uniform draw order, so
        // y_rep is bit-identical under a fixed predict RNG).
        impl_->data().set("eta",   arma::vec(N, arma::fill::zeros));
        impl_->data().set("prob",  arma::vec(N, arma::fill::value(0.5)));
        impl_->data().set("y_rep", arma::vec(N, arma::fill::zeros));

        impl_->data().register_refresher(
            "eta",
            [p](const AI4BayesCode::shared_data_t& d) {
                const arma::vec& beta_cur = d.get("beta");
                const arma::vec& X_flat   = d.get("X");
                const std::size_t N_cur   = X_flat.n_elem / p;
                arma::vec eta(N_cur);
                for (std::size_t i = 0; i < N_cur; ++i) {
                    double lin = 0.0;
                    for (std::size_t j = 0; j < p; ++j)
                        lin += X_flat[i + j * N_cur] * beta_cur[j];
                    eta[i] = lin;
                }
                return eta;
            });
        impl_->data().register_refresher(
            "prob",
            [](const AI4BayesCode::shared_data_t& d) {
                const arma::vec& eta = d.get("eta");
                arma::vec prob(eta.n_elem);
                for (std::size_t i = 0; i < eta.n_elem; ++i)
                    prob[i] = 1.0 / (1.0 + std::exp(-eta[i]));
                return prob;
            });
        impl_->data().declare_invalidates("beta", {"eta", "prob"});

        // Predict DAG: full generative chain X,beta -> eta -> prob -> y_rep.
        impl_->data().declare_predict_edges("X",    {"eta"});
        impl_->data().declare_predict_edges("beta", {"eta"});
        impl_->data().declare_predict_edges("eta",  {"prob"});
        impl_->data().declare_predict_edges("prob", {"y_rep"});
        impl_->data().declare_data_input("X");
        // beta ~ N(0, prior_sd^2): prior_sd is a ctor scalar, no
        // shared_data slot -> no context edge (const-prior rule).

        impl_->data().register_stochastic_refresher(
            "y_rep",
            [](const AI4BayesCode::shared_data_t& d,
               std::mt19937_64& rng) {
                const arma::vec& prob = d.get("prob");
                std::uniform_real_distribution<double> ud(0.0, 1.0);
                arma::vec y_rep(prob.n_elem);
                for (std::size_t i = 0; i < prob.n_elem; ++i)
                    y_rep[i] = (ud(rng) < prob[i]) ? 1.0 : 0.0;
                return y_rep;
            });

        // Single block: pg_logistic_block owns beta AND internally draws
        // ω each sweep.
        pg_logistic_block_config cfg;
        cfg.name = "beta";
        cfg.p    = p;
        cfg.y_key = "y";
        cfg.X_key = "X";
        cfg.prior_mean = arma::vec(p, arma::fill::zeros);
        cfg.prior_cov  = (prior_sd * prior_sd) *
            arma::eye<arma::mat>(p, p);
        cfg.initial_beta = arma::vec(p, arma::fill::zeros);
        cfg.n_pg_terms   = 128;
        impl_->add_child(std::make_unique<pg_logistic_block>(std::move(cfg)));

        if (keep_history_) impl_->set_keep_history(true);
    }

    // ---- Canonical 6-method R interface ----

    void step() { step(1); }              // no-arg convenience: one sweep

    void step(int n_steps) {
        if (n_steps < 0) throw std::runtime_error("n_steps must be >= 0");
        for (int i = 0; i < n_steps; ++i) impl_->step(rng_);
    }

    AI4BayesCode::state_map get_current() const {
        AI4BayesCode::state_map out;
        out["beta"] = impl_->data().get("beta");
        return out;
    }

    // X is supplied as a flat arma::vec (vectorised column-major) of length
    // N*p; Python users build it via e.g. np.asfortranarray(X).ravel().
    void set_current(const AI4BayesCode::state_map& params) {
        auto* beta_blk = dynamic_cast<pg_logistic_block*>(&impl_->child(0));
        auto it_b = params.find("beta");
        if (it_b != params.end()) {
            const arma::vec& bnew = it_b->second;
            if (bnew.n_elem != p_)
                throw std::runtime_error("set_current: beta length must equal p");
            beta_blk->set_current(bnew);
            impl_->data().set("beta", bnew);
        }
        // X / y: canonical dynamic-N pattern (codegen_cpp.md §7a).
        // p is strict (beta length); N is dynamic, derived from
        // X.n_elem / p when X is supplied.
        auto it_X = params.find("X");
        auto it_y = params.find("y");
        if (it_X != params.end()) {
            const arma::vec& X_new = it_X->second;
            if (X_new.n_elem == 0 || X_new.n_elem % p_ != 0)
                throw std::runtime_error("set_current: X length " +
                    std::to_string(X_new.n_elem) +
                    " not a positive multiple of p = " +
                    std::to_string(p_));
            const std::size_t N_new = X_new.n_elem / p_;
            if (it_y != params.end()) {
                if (it_y->second.n_elem != N_new)
                    throw std::runtime_error("set_current: X implies N=" +
                        std::to_string(N_new) + " but y has length " +
                        std::to_string(it_y->second.n_elem));
            }
            impl_->data().set("X", X_new);
            if (N_new != N_) {
                impl_->data().set("y_rep",
                                  arma::vec(N_new, arma::fill::zeros));
                if (keep_history_ && impl_->history_size() > 1) {
#ifdef AI4BAYESCODE_RCPP_MODULE
                    Rcpp::warning("set_current: N changed from %zu to %zu; "
                                  "clearing history (mixed-N is unsupported).",
                                  N_, N_new);
#endif
                    impl_->clear_history();
                }
            }
            N_ = N_new;
        }
        if (it_y != params.end()) {
            const arma::vec& y_new = it_y->second;
            if (y_new.n_elem != N_)
                throw std::runtime_error("set_current: y length " +
                    std::to_string(y_new.n_elem) +
                    " != current N = " + std::to_string(N_));
            for (std::size_t i = 0; i < N_; ++i)
                if (y_new[i] != 0.0 && y_new[i] != 1.0)
                    throw std::runtime_error("y must be 0/1");
            impl_->data().set("y", y_new);
        }
    }

    // predict_at: optional X = arma::vec (vectorised N_test*p, column-major).
    //   * keep_history = FALSE: single predict at current draw — refreshed
    //     key returned as 1 x N matrix.
    //   * keep_history = TRUE:  loops over ALL posterior draws of beta —
    //     refreshed key returned as n_draws x N matrix (posterior predictive).
    AI4BayesCode::history_map predict_at(const AI4BayesCode::state_map& new_data) const {
        // Parse optional X once (shared by both modes).
        bool has_X = false;
        arma::vec x_flat;
        auto it_X = new_data.find("X");
        if (it_X != new_data.end()) {
            x_flat = it_X->second;
            if (x_flat.n_elem % p_ != 0)
                throw std::runtime_error(
                    "predict_at: X must be vectorised N_test*p (column-major)");
            has_X = true;
        }

        AI4BayesCode::history_map out;

        if (!keep_history_) {
            // ---- Stateful mode: single predict at current draw ------------
            block_context replaced;
            if (has_X) replaced["X"] = x_flat;
            block_context result = impl_->predict_at(replaced, predict_rng_);
            for (const auto& kv : result) {
                arma::mat m(1, kv.second.n_elem);
                for (std::size_t j = 0; j < kv.second.n_elem; ++j)
                    m(0, j) = kv.second[j];
                out.emplace(kv.first, std::move(m));
            }
            return out;
        }

        // ---- History mode: loop over all posterior draws of beta ----------
        // beta IS the block name of pg_logistic_block, so framework accepts
        // replaced["beta"]. y_rep refresher reads beta + X from scratch.
        AI4BayesCode::history_map hist = impl_->get_history();
        auto it_b = hist.find("beta");
        if (it_b == hist.end()) {
            throw std::runtime_error(
                "LogisticRegression::predict_at: keep_history_ requires "
                "beta history but get_history() lacks it. Did you "
                "construct with keep_history = TRUE?");
        }
        const arma::mat& beta_hist = it_b->second;   // n_draws x p
        const std::size_t n_draws  = beta_hist.n_rows;

        std::unordered_map<std::string, std::vector<arma::vec>> collected;
        // Per-draw X is set EVERY iteration (new x_flat, or the training
        // X) so predict_at Pass-1 recomputes the deterministic chain
        // eta->prob for each historical beta draw. Without an explicit X
        // in `replaced`, Pass-1's strict availability rule treats the
        // data_input X as unavailable and would leave eta stale.
        const arma::vec X_for_draw = has_X ? x_flat : impl_->data().get("X");
        for (std::size_t d = 0; d < n_draws; ++d) {
            block_context replaced;
            replaced["X"]    = X_for_draw;
            replaced["beta"] = beta_hist.row(d).t();
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

    AI4BayesCode::dag_info get_dag() const { return impl_->get_dag(); }
    AI4BayesCode::history_map get_history() const { return impl_->get_history(); }

private:
    std::mt19937_64                  rng_;
    mutable std::mt19937_64          predict_rng_;
    std::unique_ptr<composite_block> impl_;
    bool                             keep_history_ = false;
    std::size_t                      N_ = 0;
    std::size_t                      p_ = 0;
};

#ifdef AI4BAYESCODE_RCPP_MODULE
RCPP_MODULE(LogisticRegression_module) {
    Rcpp::class_<LogisticRegression>("LogisticRegression")
        .constructor<arma::mat, arma::vec, double, int>(
            "Legacy constructor; keep_history defaults to FALSE.")
        .constructor<arma::mat, arma::vec, double, int, bool>(
            "Construct with X (N x p), y (length N, 0/1), prior_sd, seed, keep_history.")
        .method("step", (void (LogisticRegression::*)())    &LogisticRegression::step, "Run one sweep.")
        .method("step", (void (LogisticRegression::*)(int)) &LogisticRegression::step, "Run n sweeps.")
        .method("get_current", &LogisticRegression::get_current)
        .method("set_current", &LogisticRegression::set_current)
        .method("predict_at",  &LogisticRegression::predict_at)
        .method("get_dag",     &LogisticRegression::get_dag)
        .method("get_history", &LogisticRegression::get_history);
}
#endif

#ifdef AI4BAYESCODE_PYBIND_MODULE
#include "AI4BayesCode/pybind_casters.hpp"
PYBIND11_MODULE(LogisticRegression, m) {
    AI4BayesCode::register_ai4bayescode_types(m);
    pybind11::class_<LogisticRegression>(m, "LogisticRegression")
        .def(pybind11::init<arma::mat, arma::vec, double, int, bool>(),
             pybind11::arg("X"), pybind11::arg("y"),
             pybind11::arg("prior_sd") = 10.0,
             pybind11::arg("rng_seed") = 1,
             pybind11::arg("keep_history") = false,
             "Bayesian logistic regression via Polya-Gamma augmentation "
             "(Polson-Scott-Windle 2013). 10-100x faster than NUTS on "
             "logistic for moderate p.")
        .def("step", (void (LogisticRegression::*)())    &LogisticRegression::step, "Run one sweep.")
        .def("step", (void (LogisticRegression::*)(int)) &LogisticRegression::step, pybind11::arg("n_steps"))
        .def("get_current", &LogisticRegression::get_current)
        .def("set_current", &LogisticRegression::set_current, pybind11::arg("params"))
        .def("predict_at",  &LogisticRegression::predict_at, pybind11::arg("new_data"))
        .def("get_dag",     &LogisticRegression::get_dag)
        .def("get_history", &LogisticRegression::get_history);
}
#endif

#if !defined(AI4BAYESCODE_RCPP_MODULE) && !defined(AI4BAYESCODE_PYBIND_MODULE)
#include <cstdio>
// ============================================================================
//  Standalone demo: simulate Bernoulli data from a KNOWN beta, fit via
//  Polya-Gamma Gibbs, recover the posterior mean of beta, and check it
//  recovers truth AND beats the naive zero-beta baseline on log-likelihood.
// ============================================================================
namespace {

double sigmoid(double z) { return 1.0 / (1.0 + std::exp(-z)); }

// Bernoulli log-likelihood sum_i [ y_i log p_i + (1-y_i) log(1-p_i) ]
// at linear predictor X * beta.
double bernoulli_loglik(const arma::mat& X, const arma::vec& y,
                        const arma::vec& beta) {
    const arma::vec eta = X * beta;
    double ll = 0.0;
    for (std::size_t i = 0; i < y.n_elem; ++i) {
        const double p = sigmoid(eta[i]);
        const double pc = std::min(std::max(p, 1e-12), 1.0 - 1e-12);
        ll += y[i] * std::log(pc) + (1.0 - y[i]) * std::log(1.0 - pc);
    }
    return ll;
}

}  // namespace

int main() {
    // ---- Known truth -------------------------------------------------------
    const std::size_t N = 800;
    const std::size_t p = 3;
    const arma::vec beta_true = {-0.5, 1.2, -0.8};

    // ---- Simulate design + Bernoulli responses -----------------------------
    std::mt19937_64 sim_rng(2026);
    std::normal_distribution<double>      gen(0.0, 1.0);
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    arma::mat X(N, p);
    for (std::size_t i = 0; i < N; ++i) {
        X(i, 0) = 1.0;                       // intercept column
        for (std::size_t j = 1; j < p; ++j)
            X(i, j) = gen(sim_rng);
    }
    arma::vec y(N);
    for (std::size_t i = 0; i < N; ++i) {
        const double prob = sigmoid(arma::dot(X.row(i).t(), beta_true));
        y[i] = (uni(sim_rng) < prob) ? 1.0 : 0.0;
    }

    // ---- Fit: PG-augmented Gibbs -------------------------------------------
    LogisticRegression model(X, y, /*prior_sd=*/10.0, /*rng_seed=*/7,
                             /*keep_history=*/false);
    model.step(500);   // warmup

    arma::vec beta_bar(p, arma::fill::zeros);
    const int M = 2000;
    for (int s = 0; s < M; ++s) {
        model.step(1);
        const auto cur = model.get_current();
        beta_bar += cur.at("beta");
    }
    beta_bar /= static_cast<double>(M);

    // ---- Report ------------------------------------------------------------
    std::printf("LogisticRegression demo (N=%zu, p=%zu):\n", N, p);
    double max_abs_err = 0.0;
    for (std::size_t j = 0; j < p; ++j) {
        const double err = std::abs(beta_bar[j] - beta_true[j]);
        max_abs_err = std::max(max_abs_err, err);
        std::printf("  beta[%zu]  hat=%+.3f  truth=%+.3f  |err|=%.3f\n",
                    j, beta_bar[j], beta_true[j], err);
    }

    // Posterior mean must (a) be close to truth and (b) beat the naive
    // zero-beta baseline on held-in Bernoulli log-likelihood.
    const arma::vec beta_zero(p, arma::fill::zeros);
    const double ll_fit  = bernoulli_loglik(X, y, beta_bar);
    const double ll_null = bernoulli_loglik(X, y, beta_zero);
    std::printf("  loglik: fit=%.2f  null(beta=0)=%.2f  (higher is better)\n",
                ll_fit, ll_null);

    const bool ok = (max_abs_err < 0.25) && (ll_fit > ll_null);
    std::printf("%s\n",
                ok ? "[demo PASS] PG-Gibbs recovers beta and beats null"
                   : "[demo FAIL]");
    return ok ? 0 : 1;
}
#endif

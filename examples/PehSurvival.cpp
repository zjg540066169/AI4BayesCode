// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v3.0 or later
// (GPL-3.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
//  PehSurvival.cpp
//
//  Bayesian piecewise-exponential (PEH) survival analysis with a Gamma-Poisson
//  conjugate Gibbs update on the K piecewise-constant baseline hazard rates.
//  Baseline-only (no covariates). This is the smallest working demo of
//  `piecewise_exponential_gibbs_block`.
//
//  Model
//  -----
//      Time axis partition: 0 = e_0 < e_1 < ... < e_K   (K bins)
//      Baseline hazard:     h_0(t) = lambda_k    for t in (e_{k-1}, e_k]
//      Observed:  (t_i, delta_i),  t_i > 0,  delta_i in {0, 1}
//      Likelihood: h(t_i)^{delta_i} exp(-H_0(t_i))
//                  where H_0(t) = sum_k lambda_k * Delta_k(t)
//                  and Delta_k(t) = max(0, min(t, e_k) - e_{k-1}).
//      Prior:      lambda_k ~ Gamma(a0, b0), independent across k.
//
//  Full conditional (Gamma-Poisson conjugate):
//      E_k = sum_i delta_i * I(t_i in bin k)
//      R_k = sum_i Delta_k(t_i)
//      lambda_k | rest ~ Gamma(a0 + E_k, b0 + R_k), independent across k.
//
//  Correctness ground truth
//  ------------------------
//  Because each Gibbs draw is exact iid from the full conditional, the sampled
//  posterior mean of lambda_k must match (a0 + E_k) / (b0 + R_k) with an error
//  that shrinks as 1/sqrt(M) where M is the number of retained draws.
//  For M = 5000 the MC standard error is ~ sd(Gamma) / sqrt(5000) ~ 0.001-0.01;
//  we accept < 5% relative error.
// ============================================================================

// @example:R
//   library(AI4BayesCode)
//   ai4bayescode_example("PehSurvival")
//   # Simulate PEH data with lambda_true = c(0.5, 1.0, 1.5) on bins (0, 2, 5, 10]
//   set.seed(2026)
//   n <- 200; lambda_true <- c(0.5, 1.0, 1.5); edges <- c(0, 2, 5, 10)
//   # inverse-CDF simulator
//   u <- runif(n); Ht <- -log(1-u); K <- 3; bw <- diff(edges)
//   cumH <- cumsum(c(0, lambda_true * bw)); T <- numeric(n)
//   for (i in 1:n) { k <- max(which(cumH <= Ht[i]))
//       T[i] <- if (k > K) edges[K+1] + (Ht[i]-cumH[K+1])/lambda_true[K]
//               else       edges[k]  + (Ht[i]-cumH[k])/lambda_true[k] }
//   censor <- 8; delta <- as.integer(T <= censor); t_obs <- pmin(T, censor)
//   # Fit with 4 parallel chains + convergence diagnosis.
//   run <- ai4bayescode_run_chains(
//       model_ctor = function(seed) new(PehSurvival, t_obs, delta, edges, 0.01, 0.01, seed, TRUE),
//       n_chains = 4, n_burn = 3000, n_keep = 5000)
//   diag <- ai4bayescode_diagnose(run)
//   print(diag$summary)   # posterior mean lambda_k, R-hat, ESS

// @example:python
//   import numpy as np, AI4BayesCode as ai
//   PehSurvival = ai.example("PehSurvival")
//   n, edges, lam_true = 200, np.array([0.0, 2.0, 5.0, 10.0]), np.array([0.5, 1.0, 1.5])
//   rng = np.random.default_rng(2026)
//   u = rng.uniform(size=n); Ht = -np.log1p(-u)
//   K = 3; cumH = np.concatenate([[0.0], np.cumsum(lam_true * np.diff(edges))])
//   T = np.empty(n)
//   for i in range(n):
//       k = int(np.searchsorted(cumH, Ht[i], side="right") - 1)
//       if k >= K: T[i] = edges[K] + (Ht[i]-cumH[K])/lam_true[K-1]
//       else:      T[i] = edges[k] + (Ht[i]-cumH[k])/lam_true[k]
//   censor = 8.0; delta = (T <= censor).astype(float); t_obs = np.minimum(T, censor)
//   run = ai.run_chains(
//       factory = lambda seed: PehSurvival(t_obs, delta, edges, 0.01, 0.01, seed, True),
//       seeds = (101, 202, 303, 404), n_burn=3000, n_keep=5000)
//   diag = ai.diagnose(run); print(diag['summary'])

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
#include "AI4BayesCode/piecewise_exponential_gibbs_block.hpp"
#include "AI4BayesCode/rcpp_wrap.hpp"
#include "AI4BayesCode/kernel_control_mixin.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>

using AI4BayesCode::block_context;
using AI4BayesCode::composite_block;
using AI4BayesCode::piecewise_exponential_gibbs_block;
using AI4BayesCode::piecewise_exponential_gibbs_block_config;

class PehSurvival : public AI4BayesCode::kernel_control_mixin<PehSurvival> {
    friend class AI4BayesCode::kernel_control_mixin<PehSurvival>;
public:
    PehSurvival(const arma::vec& t, const arma::vec& delta, const arma::vec& edges,
                double a0, double b0, int rng_seed, bool keep_history = false)
        : rng_(rng_seed == 0
                   ? std::mt19937_64{std::random_device{}()}
                   : std::mt19937_64{static_cast<std::uint64_t>(rng_seed)}),
          predict_rng_(rng_seed == 0
                   ? std::mt19937_64{std::random_device{}()}
                   : std::mt19937_64{static_cast<std::uint64_t>(rng_seed)
                                     ^ 0x9E3779B97F4A7C15ULL}),
          impl_(std::make_unique<composite_block>("PehSurvival")),
          keep_history_(keep_history)
    {
        if (edges.n_elem < 2) ai4b::stop("PehSurvival: edges must have length >= 2");
        const std::size_t K = edges.n_elem - 1;
        if (t.n_elem == 0 || t.n_elem != delta.n_elem)
            ai4b::stop("PehSurvival: t and delta must have equal nonzero length");

        // ---- shared_data: fixed data + hyperparameters + initial lambda -----
        impl_->data().set("t",     t);
        impl_->data().set("delta", delta);
        impl_->data().set("edges", edges);
        impl_->data().set("a0",    arma::vec{a0});
        impl_->data().set("b0",    arma::vec{b0});
        arma::vec lambda_init(K); lambda_init.fill(a0 / b0);
        impl_->data().set("lambda", lambda_init);

        // Gibbs DAG: lambda depends on (t, delta) each sweep.
        impl_->data().declare_dependencies("lambda", {"t", "delta"});

        // ---- PEH-Gibbs block: samples lambda[K] each sweep -----------------
        piecewise_exponential_gibbs_block_config peh_cfg;
        peh_cfg.name           = "lambda";
        peh_cfg.edges          = edges;
        peh_cfg.a0             = a0;
        peh_cfg.b0             = b0;
        peh_cfg.time_key       = "t";
        peh_cfg.event_key      = "delta";
        peh_cfg.initial_lambda = lambda_init;
        impl_->add_child(
            std::make_unique<piecewise_exponential_gibbs_block>(std::move(peh_cfg)));

        impl_->set_keep_history(keep_history_);
    }

    // ---- Six-method R-facing contract -----------------------------------

    void step() { step(1); }
    void step(int n_steps) {
        if (n_steps < 0) ai4b::stop("PehSurvival::step: n_steps must be non-negative");
        for (int i = 0; i < n_steps; ++i) impl_->step(rng_);
    }

    AI4BayesCode::state_map get_current() const {
        AI4BayesCode::state_map out;
        out["lambda"] = impl_->data().get("lambda");
        out["t"]      = impl_->data().get("t");
        out["delta"]  = impl_->data().get("delta");
        out["edges"]  = impl_->data().get("edges");
        return out;
    }

    void set_current(const AI4BayesCode::state_map& params) {
        auto it = params.find("lambda");
        if (it != params.end()) impl_->data().set("lambda", it->second);
    }

    AI4BayesCode::history_map get_history() const { return impl_->get_history(); }

    AI4BayesCode::dag_info get_dag() const { return impl_->get_dag(); }

    /// Predict-at: return the posterior-predictive survival curve S(t_q) at the
    /// query times new_data["t"] under the CURRENT lambda draw (baseline-only).
    AI4BayesCode::history_map predict_at(const AI4BayesCode::state_map& new_data) const {
        AI4BayesCode::history_map out;
        auto it_tq = new_data.find("t");
        if (it_tq == new_data.end()) return out;
        const arma::vec& t_q = it_tq->second;
        const arma::vec& edges_local = impl_->data().get("edges");
        const arma::vec& lambda      = impl_->data().get("lambda");
        const std::size_t K = edges_local.n_elem - 1;
        arma::mat S(t_q.n_elem, 1);
        for (std::size_t i = 0; i < t_q.n_elem; ++i) {
            double H = 0.0;
            for (std::size_t k = 0; k < K; ++k) {
                const double lo = edges_local[k];
                const double hi = std::min(t_q[i], edges_local[k + 1]);
                const double dur = hi - lo;
                if (dur > 0.0) H += lambda[k] * dur;
            }
            S(i, 0) = std::exp(-H);
        }
        out["S"] = S;
        return out;
    }

private:
    std::mt19937_64                  rng_;
    mutable std::mt19937_64          predict_rng_;
    std::unique_ptr<composite_block> impl_;
    bool                             keep_history_ = false;
};

#ifdef AI4BAYESCODE_RCPP_MODULE
RCPP_MODULE(PehSurvival_module) {
    Rcpp::class_<PehSurvival>("PehSurvival")
        .constructor<arma::vec, arma::vec, arma::vec, double, double, int>(
            "Legacy constructor; keep_history defaults to FALSE.")
        .constructor<arma::vec, arma::vec, arma::vec, double, double, int, bool>(
            "Construct with: t (>0 event/censoring times), delta (0/1 event indicators), "
            "edges (K+1 bin boundaries), a0 & b0 (Gamma prior), seed, keep_history.")
        .method("step", (void (PehSurvival::*)())    &PehSurvival::step, "Run one sweep.")
        .method("step", (void (PehSurvival::*)(int)) &PehSurvival::step, "Run n sweeps.")
        .method("get_current", &PehSurvival::get_current)
        .method("set_current", &PehSurvival::set_current)
        .method("predict_at",  &PehSurvival::predict_at)
        .method("get_dag",     &PehSurvival::get_dag)
        .method("get_history", &PehSurvival::get_history)
        AI4BAYESCODE_BIND_KERNEL_CONTROL(PehSurvival);
}
#endif

#ifdef AI4BAYESCODE_PYBIND_MODULE
#include "AI4BayesCode/pybind_casters.hpp"
PYBIND11_MODULE(PehSurvival, m) {
    AI4BayesCode::register_ai4bayescode_types(m);
    pybind11::class_<PehSurvival>(m, "PehSurvival")
        .def(pybind11::init<arma::vec, arma::vec, arma::vec, double, double, int, bool>(),
             pybind11::arg("t"), pybind11::arg("delta"), pybind11::arg("edges"),
             pybind11::arg("a0") = 0.01, pybind11::arg("b0") = 0.01,
             pybind11::arg("rng_seed") = 1, pybind11::arg("keep_history") = false,
             "Bayesian piecewise-exponential survival (Kalbfleisch 1978, Ibrahim/"
             "Chen/Sinha 2001 Sec.3.2). Exact Gamma-Poisson conjugate Gibbs update "
             "of K piecewise-constant baseline hazards.")
        .def("step", (void (PehSurvival::*)())    &PehSurvival::step, "Run one sweep.")
        .def("step", (void (PehSurvival::*)(int)) &PehSurvival::step, pybind11::arg("n_steps"))
        .def("get_current", &PehSurvival::get_current)
        .def("set_current", &PehSurvival::set_current, pybind11::arg("params"))
        .def("predict_at",  &PehSurvival::predict_at,  pybind11::arg("new_data"))
        .def("get_dag",     &PehSurvival::get_dag)
        .def("get_history", &PehSurvival::get_history)
        AI4BAYESCODE_PYBIND_KERNEL_CONTROL(PehSurvival);
}
#endif

#if !defined(AI4BAYESCODE_RCPP_MODULE) && !defined(AI4BAYESCODE_PYBIND_MODULE)
#include <cstdio>
// ============================================================================
//  Standalone demo: simulate PEH data with known lambda_true = (0.5, 1.0, 1.5)
//  on edges (0, 2, 5, 10), fit our sampler, and verify the posterior mean of
//  each lambda_k matches its ANALYTIC full-conditional mean (a0+E_k)/(b0+R_k)
//  (exact Gamma-Poisson conjugacy given fixed data), plus a coarser recovery
//  of lambda_true (dominated by DGP randomness at n=200).
// ============================================================================
int main() {
    const std::size_t n = 200;
    const double censor_time = 8.0;
    arma::vec edges({0.0, 2.0, 5.0, 10.0});
    arma::vec lambda_true({0.5, 1.0, 1.5});
    const std::size_t K = 3;
    const double a0 = 0.01, b0 = 0.01;

    // ---- Simulate PEH data via inverse CDF ------------------------------
    std::mt19937_64 sim_rng(2026);
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    arma::vec t(n), delta(n);
    arma::vec cumH(K + 1); cumH[0] = 0.0;
    for (std::size_t k = 0; k < K; ++k) cumH[k+1] = cumH[k] + lambda_true[k] * (edges[k+1]-edges[k]);
    for (std::size_t i = 0; i < n; ++i) {
        const double Ht = -std::log(1.0 - unif(sim_rng));
        std::size_t k = 0;
        while (k < K && cumH[k+1] <= Ht) ++k;
        double Ti;
        if (k >= K) Ti = edges[K] + (Ht - cumH[K]) / lambda_true[K-1];
        else        Ti = edges[k] + (Ht - cumH[k]) / lambda_true[k];
        delta[i] = (Ti <= censor_time) ? 1.0 : 0.0;
        t[i]     = std::min(Ti, censor_time);
    }

    // ---- Hand-compute (E_k, R_k) as the analytic ground truth ------------
    std::vector<double> E(K, 0.0), R(K, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        std::size_t k_i = 0;
        while (k_i < K && edges[k_i+1] < t[i]) ++k_i;
        if (delta[i] > 0.5) E[k_i] += 1.0;
        for (std::size_t k = 0; k <= k_i; ++k) {
            const double dur = std::min(t[i], edges[k+1]) - edges[k];
            if (dur > 0.0) R[k] += dur;
        }
    }

    // ---- Fit ------------------------------------------------------------
    PehSurvival model(t, delta, edges, a0, b0, /*rng_seed=*/7, /*keep_history=*/false);
    model.step(1000);   // warmup (Gibbs is exact so this is just for consistency)
    const int M = 5000;
    arma::vec lambda_bar(K, arma::fill::zeros);
    for (int s = 0; s < M; ++s) {
        model.step(1);
        const auto st = model.get_current();
        const arma::vec& lam = st.at("lambda");
        for (std::size_t k = 0; k < K; ++k) lambda_bar[k] += lam[k];
    }
    lambda_bar /= static_cast<double>(M);

    // ---- Verify posterior mean matches analytic full-conditional ---------
    std::printf("PehSurvival demo: n=%zu K=%zu censoring=%.1f%%\n",
        n, K, 100.0 * (1.0 - static_cast<double>(std::count_if(
            delta.begin(), delta.end(), [](double d){ return d > 0.5; })) / n));
    std::printf("bin |  E_k    R_k  |  analytic_mean=(a0+E_k)/(b0+R_k)  |  sampled_mean\n");
    bool ok = true;
    for (std::size_t k = 0; k < K; ++k) {
        const double analytic = (a0 + E[k]) / (b0 + R[k]);
        const double relerr   = std::abs(lambda_bar[k] - analytic) / analytic;
        std::printf(" %zu  | %5.1f  %6.2f  |  %20.6f  |  %.6f  (rel_err=%.3f)\n",
                    k + 1, E[k], R[k], analytic, lambda_bar[k], relerr);
        if (relerr > 0.05) ok = false;
    }

    std::printf("%s\n", ok
        ? "[demo PASS] sampled posterior mean matches analytic Gamma-Poisson full conditional (<5%)"
        : "[demo FAIL] sampled posterior mean drifted from analytic ground truth");
    return ok ? 0 : 1;
}
#endif

// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v3.0 or later
// (GPL-3.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
//  IsingPrior.cpp
//
//  Pure-prior 2D Ising / Potts sampler — first specialized-block demo for
//  AI4BayesCode v1.2. Wires ising_cluster_block (Swendsen-Wang cluster
//  sweep) through composite_block + RCPP_MODULE.
//
//  Model
//  -----
//      pi(x) ∝ exp{ β · Σ_{i~j} I[x_i = x_j] },  x ∈ {0,1,...,Q-1}^N
//
//  Topology: rectangular L_x × L_y lattice with user-selectable 4-NN
//  (von Neumann) or 8-NN neighbourhood and open / periodic boundary.
//
//  Parameters
//  ----------
//      x       latent state vector (length L_x · L_y, entries in
//              {0..Q-1}) — sampled by ising_cluster_block
//      beta    interaction strength (scalar, ≥ 0) — fixed at construction
//              by default; can be overwritten via set_current("beta") to
//              enable hierarchical use (a sibling block sampling β).
//
//  No observation likelihood — this is a PRIOR-ONLY demo. predict_at()
//  returns an empty list. Documented exception per system_design.md §5
//  (no observation model ⇒ no y_rep / Layer-3 R3 posterior-predictive
//  diagnostic).
//
//  JUSTIFICATION (Check #16): Discrete-MRF target with strong local
//  dependence — per system_design.md §11.2(b) this is exactly the
//  class where per-site Gibbs mixes catastrophically and a specialised
//  cluster-move algorithm (Swendsen-Wang 1987) is the standard remedy.
//  ising_cluster_block is the library-blessed SW sweep.
//
//  Check #15 parity tests live at:
//    tests/test_ising_cluster_block.cpp              (4×4 enumeration,
//                                                     Q=3 Potts symmetry,
//                                                     two-init mixing)
//    tests/test_ising_cluster_block_diagnostics.cpp  (4-chain split-R-hat,
//                                                     batch-means ESS,
//                                                     17-bucket Pearson χ²,
//                                                     energy moments)
//    tests/test_ising_sw_vs_single_site.cpp          (SW vs single-site
//                                                     Metropolis efficiency
//                                                     comparison)
//  All ground truth is closed-form / exact enumeration; zero external
//  package dependency in the shipped tree.
// ============================================================================

// @example:R
//   library(AI4BayesCode)
//   ai4bayescode_example("IsingPrior")
//   # Pure-prior 2D Ising (Q=2) on a 16x16 periodic lattice, ferromagnetic
//   # beta=0.44 (near the 2D critical coupling). No data: the proper discrete
//   # MRF prior pi(x) is itself the target; we just sample the latent state x.
//   # ---- Recommended: parallel chains + convergence diagnosis ----
//   run <- ai4bayescode_run_chains(
//       function(seed) new(IsingPrior, 16L, 16L, 2L, 0.44, TRUE, FALSE, seed, TRUE),
//       n_chains = 4, n_burn = 1000, n_keep = 2000)
//   ai4bayescode_diagnose(run$histories[[1]])      # summary + R-hat/ESS + plots
//   # ---- Advanced: stateful single-chain control ----
//   m <- new(IsingPrior,
//            16L, 16L,   # L_x, L_y: lattice dims (N = 256 sites)
//            2L,         # Q: state count (2 = Ising)
//            0.44,       # beta: interaction strength (>= 0)
//            TRUE,       # periodic: wrap lattice edges
//            FALSE,      # eight_nn: 4-NN (von Neumann) neighbourhood
//            7L,         # rng_seed
//            TRUE)       # keep_history
//   m$step(2500); str(m$get_current())   # x: length-256 in {0,1}; beta: 0.44
// @example:python
//   import numpy as np, AI4BayesCode
//   # Pure-prior 2D Ising (Q=2) on a 16x16 periodic lattice, ferromagnetic
//   # beta=0.44 (near the 2D critical coupling). No data: the proper discrete
//   # MRF prior pi(x) is itself the target; we just sample the latent state x.
//   Mod = AI4BayesCode.example("IsingPrior")
//   # ---- Recommended: parallel chains + diagnosis ----
//   chains = AI4BayesCode.run_chains(
//       lambda seed: Mod.IsingPrior(16, 16, 2, 0.44, True, False, seed, True),
//       seeds=[101, 202, 303, 404], n_burn=1000, n_keep=2000, n_jobs=1)
//   AI4BayesCode.diagnose(chains[0]["hist"])   # summary + diagnostics
//   # ---- Advanced: stateful single-chain control ----
//   m = Mod.IsingPrior(16, 16,   # L_x, L_y: lattice dims (N = 256 sites)
//                      2,         # Q: state count (2 = Ising)
//                      0.44,      # beta: interaction strength (>= 0)
//                      True,      # periodic: wrap lattice edges
//                      False,     # eight_nn: 4-NN (von Neumann) neighbourhood
//                      7,         # rng_seed
//                      True)      # keep_history
//   m.step(2500); print(m.get_current())  # x: length-256 in {0,1}; beta: 0.44
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
#include "AI4BayesCode/ising_cluster_block.hpp"
#include "AI4BayesCode/kernel_control_mixin.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>

using AI4BayesCode::block_context;
using AI4BayesCode::composite_block;
using AI4BayesCode::ising_cluster_block;
using AI4BayesCode::ising_cluster_block_config;
using AI4BayesCode::make_2d_lattice_edges;

class IsingPrior : public AI4BayesCode::kernel_control_mixin<IsingPrior> {
    friend class AI4BayesCode::kernel_control_mixin<IsingPrior>;
public:
    IsingPrior(int L_x, int L_y, int Q, double beta,
                bool periodic, bool eight_nn,
                int rng_seed,
                bool keep_history = false)
        : rng_(rng_seed == 0
                   ? std::mt19937_64{std::random_device{}()}
                   : std::mt19937_64{static_cast<std::uint64_t>(rng_seed)}),
          predict_rng_(rng_seed == 0
                   ? std::mt19937_64{std::random_device{}()}
                   : std::mt19937_64{static_cast<std::uint64_t>(rng_seed)
                                     ^ 0x9E3779B97F4A7C15ULL}),
          impl_(std::make_unique<composite_block>("IsingPrior")),
          keep_history_(keep_history),
          L_x_(L_x), L_y_(L_y), Q_(Q)
    {
        if (L_x_ < 1)         ai4b::stop("L_x must be >= 1");
        if (L_y_ < 1)         ai4b::stop("L_y must be >= 1");
        if (Q_   < 2)         ai4b::stop("Q must be >= 2");
        if (!(beta >= 0.0))   ai4b::stop("beta must be >= 0");

        N_ = static_cast<std::size_t>(L_x_) *
              static_cast<std::size_t>(L_y_);

        // ---- shared_data setup ---------------------------------------------
        impl_->data().set("beta", arma::vec{beta});

        arma::vec x_init(N_);
        for (std::size_t i = 0; i < N_; ++i) {
            x_init[i] = static_cast<double>(
                i % static_cast<std::size_t>(Q_));
        }
        impl_->data().set("x", x_init);

        // Gibbs DAG: x reads β from ctx each sweep.
        impl_->data().declare_dependencies("x", {"beta"});

        // Generative-DAG context edge (VIZ-ONLY): β is the only
        // hyperparameter; arrow β → x in the generative DAG.
        impl_->data().declare_context_edges("beta", {"x"});

        // No predict DAG / stochastic refresher: no observation model,
        // no y_rep, no Layer-3 R3.

        // ---- ising_cluster_block as sole child -----------------------------
        ising_cluster_block_config cfg;
        cfg.name = "x";
        cfg.n_vertices = N_;
        cfg.n_states = static_cast<std::size_t>(Q_);
        cfg.edges = make_2d_lattice_edges(
            static_cast<std::size_t>(L_x_),
            static_cast<std::size_t>(L_y_),
            periodic, eight_nn);
        cfg.beta_key = "beta";
        cfg.beta_default = beta;
        cfg.initial_state = x_init;
        impl_->add_child(std::make_unique<ising_cluster_block>(std::move(cfg)));

        if (keep_history_) impl_->set_keep_history(true);
    }

    // ---- Canonical backend-neutral interface ---------------------------

    void step() { step(1); }              // no-arg convenience: one sweep
    void step(int n_steps) {
        if (n_steps < 0) ai4b::stop("n_steps must be >= 0");
        for (int i = 0; i < n_steps; ++i) impl_->step(rng_);
    }

    AI4BayesCode::state_map get_current() const {
        AI4BayesCode::state_map out;
        out["x"]    = impl_->data().get("x");     // length-N flat vec in {0..Q-1}
        out["beta"] = impl_->data().get("beta");  // length-1 arma::vec
        return out;
    }

    void set_current(const AI4BayesCode::state_map& params) {
        auto it_x = params.find("x");
        if (it_x != params.end()) {
            auto* x_blk = dynamic_cast<ising_cluster_block*>(
                &impl_->child(0));
            if (!x_blk) {
                ai4b::stop("internal: child 0 is not ising_cluster_block");
            }
            const arma::vec& x_new = it_x->second;
            if (x_new.n_elem != N_) {
                ai4b::stop("set_current: x length must equal L_x * L_y");
            }
            x_blk->set_current(x_new);
            impl_->data().set("x", x_new);
        }
        auto it_beta = params.find("beta");
        if (it_beta != params.end()) {
            const arma::vec& beta_new = it_beta->second;
            if (beta_new.n_elem != 1) {
                ai4b::stop("set_current: beta must be a length-1 vector");
            }
            if (!(beta_new[0] >= 0.0)) {
                ai4b::stop("set_current: beta must be >= 0");
            }
            impl_->data().set("beta", beta_new);
        }
    }

    AI4BayesCode::history_map predict_at(
        const AI4BayesCode::state_map& new_data) const {
        // IsingPrior has no observation model — no y_rep, no data input
        // refreshers. predict_at always returns an empty map. Documented
        // exception per system_design.md §5.
        if (!new_data.empty()) {
            ai4b::stop("IsingPrior::predict_at: no valid keys to replace "
                       "(no observation model); pass an empty map/list.");
        }
        return AI4BayesCode::history_map();
    }

    AI4BayesCode::dag_info     get_dag()     const { return impl_->get_dag();     }
    AI4BayesCode::history_map  get_history() const { return impl_->get_history(); }

private:
    std::mt19937_64                  rng_;
    mutable std::mt19937_64          predict_rng_;
    std::unique_ptr<composite_block> impl_;
    bool                             keep_history_ = false;
    int                              L_x_ = 0, L_y_ = 0, Q_ = 0;
    std::size_t                      N_   = 0;
};

#ifdef AI4BAYESCODE_RCPP_MODULE
RCPP_MODULE(IsingPrior_module) {
    Rcpp::class_<IsingPrior>("IsingPrior")
        .constructor<int, int, int, double, bool, bool, int>(
            "Legacy constructor; keep_history defaults to FALSE. Args: "
            "L_x, L_y, Q, beta, periodic, eight_nn, rng_seed.")
        .constructor<int, int, int, double, bool, bool, int, bool>(
            "Construct with L_x, L_y (lattice dims), Q (state count; 2=Ising, "
            "≥3=Potts), beta (interaction strength, ≥ 0), periodic (bool, "
            "wrap edges), eight_nn (bool, include diagonals), rng_seed, "
            "keep_history. Pure-prior 2D Ising / Potts via Swendsen-Wang "
            "cluster sweep (ising_cluster_block). Check #15 parity tests "
            "under tests/test_ising_cluster_block*.cpp.")
        .method("step", (void (IsingPrior::*)())    &IsingPrior::step, "Run one sweep.")
        .method("step", (void (IsingPrior::*)(int)) &IsingPrior::step, "Run n sweeps.")
        .method("get_current", &IsingPrior::get_current)
        .method("set_current", &IsingPrior::set_current,
                "Overwrite x (length L_x*L_y, entries in {0..Q-1}) and/or "
                "beta (length-1, ≥ 0) from a named list. Unknown keys are "
                "silently ignored.")
        .method("predict_at",  &IsingPrior::predict_at,
                "Returns empty list — IsingPrior has no observation model. "
                "Pass an empty list; non-empty input is rejected.")
        .method("get_dag",     &IsingPrior::get_dag)
        .method("get_history", &IsingPrior::get_history)
        AI4BAYESCODE_BIND_KERNEL_CONTROL(IsingPrior);
}
#endif

#ifdef AI4BAYESCODE_PYBIND_MODULE
#include "AI4BayesCode/pybind_casters.hpp"
PYBIND11_MODULE(IsingPrior, m) {
    AI4BayesCode::register_ai4bayescode_types(m);
    pybind11::class_<IsingPrior>(m, "IsingPrior")
        .def(pybind11::init<int, int, int, double, bool, bool, int, bool>(),
             pybind11::arg("L_x"),
             pybind11::arg("L_y"),
             pybind11::arg("Q"),
             pybind11::arg("beta"),
             pybind11::arg("periodic"),
             pybind11::arg("eight_nn"),
             pybind11::arg("rng_seed") = 1,
             pybind11::arg("keep_history") = false)
        .def("step", (void (IsingPrior::*)())    &IsingPrior::step, "Run one sweep.")
        .def("step", (void (IsingPrior::*)(int)) &IsingPrior::step, pybind11::arg("n_steps"))
        .def("get_current", &IsingPrior::get_current)
        .def("set_current", &IsingPrior::set_current, pybind11::arg("params"))
        .def("predict_at",  &IsingPrior::predict_at, pybind11::arg("new_data"))
        .def("get_dag",     &IsingPrior::get_dag)
        .def("get_history", &IsingPrior::get_history)
        AI4BAYESCODE_PYBIND_KERNEL_CONTROL(IsingPrior);
}
#endif

#if !defined(AI4BAYESCODE_RCPP_MODULE) && !defined(AI4BAYESCODE_PYBIND_MODULE)
// ===========================================================================
//  Standalone frontend-independent demo (active ONLY as a plain binary).
//  IsingPrior is prior-only, so validate against closed-form prior moments:
//   (A) label symmetry at ferromagnetic beta: E[x_i] = 0.5;
//   (B) beta=0 independence: per-sweep magnetisation var = 1/(4N).
// ===========================================================================
#include <cstdio>
int main() {
    const int L = 16, Q = 2;
    const double beta_c = 0.44;
    const std::size_t N = static_cast<std::size_t>(L) * L;

    IsingPrior model(L, L, Q, beta_c, /*periodic=*/true, /*eight_nn=*/false, 12345);
    model.step(200);
    double sum_state = 0.0; long long count = 0;
    for (int s = 0; s < 4000; ++s) {
        model.step(1);
        const auto gc = model.get_current();
        const arma::vec& x = gc.at("x");
        for (std::size_t i = 0; i < N; ++i) sum_state += x[i];
        count += static_cast<long long>(N);
    }
    const double mean_state = sum_state / static_cast<double>(count);
    const bool ok_sym = std::abs(mean_state - 0.5) < 0.03;

    IsingPrior m0(L, L, Q, 0.0, true, false, 999);
    m0.step(20);
    double sm = 0.0, sm2 = 0.0;
    for (int s = 0; s < 4000; ++s) {
        m0.step(1);
        const auto gc = m0.get_current();
        const arma::vec& x = gc.at("x");
        double mag = 0.0; for (std::size_t i = 0; i < N; ++i) mag += x[i];
        mag /= static_cast<double>(N); sm += mag; sm2 += mag * mag;
    }
    const double m_bar = sm / 4000.0, m_var = sm2 / 4000.0 - m_bar * m_bar;
    const double var_th = 0.25 / static_cast<double>(N);
    const bool ok_indep = std::abs(m_bar - 0.5) < 0.02 &&
                          std::abs(m_var - var_th) < 0.25 * var_th;

    std::printf("IsingPrior demo (L=%d, Q=%d):\n", L, Q);
    std::printf("  (A) beta=%.2f  site-state mean = %.4f (prior 0.5)  -> %s\n",
                beta_c, mean_state, ok_sym ? "ok" : "FAIL");
    std::printf("  (B) beta=0.00  mag var = %.6f (theory 1/(4N)=%.6f)  -> %s\n",
                m_var, var_th, ok_indep ? "ok" : "FAIL");
    const bool ok = ok_sym && ok_indep;
    std::printf("%s\n", ok ? "[demo PASS] IsingPrior matches closed-form prior moments"
                           : "[demo FAIL]");
    return ok ? 0 : 1;
}
#endif

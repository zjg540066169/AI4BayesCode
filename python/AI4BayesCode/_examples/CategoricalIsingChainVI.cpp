// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
//  CategoricalIsingChainVI.cpp
//
//  Tier A demo for v1.2 Block 4 (mean_field_categorical_vi_block).
//
//  Model
//  -----
//  Discrete Potts / Ising chain on n nodes with K states per node and
//  optional per-node external field h_i (length n):
//
//      log p~(z_1,...,z_n) = beta * sum_{(i,i+1)} I[z_i = z_{i+1}]
//                          + sum_i h_i * I[z_i = 1]
//
//  Mean-field VI factorises q(z) = prod_i Categorical(z_i ; phi_i) and
//  optimises the variational marginals phi_i in (K-1)-simplex via
//  RAABBVI on the ELBO. For small n*K (state space prod_i K_i ≤
//  exact_state_cap), gradients are EXACT; otherwise Monte Carlo with
//  configurable S.
//
//  This is a PURE-VI demo: no observation model. predict_at(list(n_draws=N))
//  returns N fresh q-samples (matrix of integer indices); empty input is
//  also valid (no-op stochastic refresher).
//
//  Validation
//  ----------
//  Cross-validation lives at tests/test_mean_field_categorical_vi_chain.cpp
//  (V1-V8: KL/TV vs 81-state exact enumeration on symmetric AND
//  asymmetric chain; PSIS-k̂ diagnostic).
//
//  JUSTIFICATION (Check #16): Discrete latents with strong local
//  dependence — system_design.md §11.2(b). Per-site Gibbs mixes
//  catastrophically near critical coupling; categorical mean-field VI
//  gives a deterministic deterministic approximation that converges
//  cleanly to a (biased) joint with correct marginals in many regimes,
//  validated by V1-V8. The textbook MF underestimate-of-joint-variance
//  caveat (Bishop §10.1.2) is correctly diagnosed by PSIS-k̂.
//
//  Reference: Bishop PRML §10.1, Jaakkola-Jordan 1999 (QMR-DT), Welandawe
//  et al. 2022 (RAABBVI).
//
//  BACKEND-NEUTRAL STATE (state_map / history_map)
//  -----------------------------------------------
//  get_current() returns a backend-neutral state_map (dict[str, np.ndarray]
//  in Python; named list in R). Because state_map values are arma::vec, the
//  n×K variational marginal matrix phi is flattened COLUMN-MAJOR into a flat
//  length-(n*K) vector under key "phi" (reshape with R matrix(phi, n, K) /
//  numpy phi.reshape(n, K, order="F")). Scalars (elbo, converged, epoch) are
//  returned as length-1 vectors. set_current() accepts phi as either the flat
//  length-(n*K) column-major vector or — under the R backend only, via a
//  named-list helper — an n×K matrix; under the backend-neutral path it is the
//  flat column-major vector. predict_at() returns a history_map with key
//  "z_samples" = an (n_draws × n) integer-valued matrix.
//
// @example:R
//   library(AI4BayesCode)
//   n <- 4L                                  # chain length (>= 2)
//   K <- 3L                                  # states per node (K^n = 81 <= 4096 -> EXACT gradients)
//   beta <- 0.8                              # ferromagnetic coupling (>= 0)
//   h <- rep(2.0, n)                         # strong external field on state 1 -> identifies a non-uniform q
//   m <- new(CategoricalIsingChainVI, n, K, beta, h, TRUE, 20260621L, TRUE)
//                                            # args: n_nodes, K, beta, h, exact_enumeration, rng_seed, keep_history
//   m$step(6000L)                            # run RAABBVI to convergence (no-op once converged)
//   cur <- m$get_current()                   # list(phi (flat n*K col-major), z, beta, h, elbo, converged, epoch)
//   matrix(cur$phi, n, K)                     # reshape phi back to n x K
// @example:python
//   import numpy as np, AI4BayesCode
//   n, K = 4, 3                              # chain length (>= 2), states per node (K^n = 81 <= 4096 -> EXACT)
//   beta = 0.8                               # ferromagnetic coupling (>= 0)
//   h = np.full(n, 2.0)                      # strong field on state 1 -> identifies a non-uniform q
//   Mod = AI4BayesCode.source("CategoricalIsingChainVI.cpp")
//   m = Mod.CategoricalIsingChainVI(n, K, beta, h, True, 20260621, True)
//   #   args: n_nodes, K, beta, h, exact_enumeration, rng_seed, keep_history
//   m.step(6000)                             # run RAABBVI to convergence
//   cur = m.get_current()                    # dict: phi (flat n*K col-major), z, beta, h, elbo, converged, epoch
//   phi = np.asarray(cur["phi"]).reshape(n, K, order="F")  # back to n x K
//   print(phi)
// @example:end
// ============================================================================

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
#include "AI4BayesCode/mean_field_categorical_vi_block.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

using AI4BayesCode::block_context;
using AI4BayesCode::composite_block;
using AI4BayesCode::mean_field_categorical_vi_block;
using AI4BayesCode::mean_field_categorical_vi_block_config;

// ============================================================================
//  User-facing class. Defined backend-neutrally (NO Rcpp/pybind types in the
//  class body) so it can be bound by BOTH the Rcpp module and the pybind
//  module below.
// ============================================================================

class CategoricalIsingChainVI {
public:
    CategoricalIsingChainVI(int n_nodes, int K, double beta,
                              const arma::vec& h_input,
                              bool exact_enumeration,
                              int rng_seed,
                              bool keep_history = false)
        : rng_(rng_seed == 0
                   ? std::mt19937_64{std::random_device{}()}
                   : std::mt19937_64{static_cast<std::uint64_t>(rng_seed)}),
          predict_rng_(rng_seed == 0
                   ? std::mt19937_64{std::random_device{}()}
                   : std::mt19937_64{static_cast<std::uint64_t>(rng_seed)
                                     ^ 0x9E3779B97F4A7C15ULL}),
          impl_(std::make_unique<composite_block>("CategoricalIsingChainVI")),
          keep_history_(keep_history),
          n_(static_cast<std::size_t>(n_nodes)),
          K_(static_cast<std::size_t>(K)),
          beta_(beta)
    {
        if (n_nodes < 2)         ai4b::stop("n_nodes must be >= 2");
        if (K < 2)               ai4b::stop("K must be >= 2");
        if (!(beta >= 0.0))      ai4b::stop("beta must be >= 0");

        // External field h: pad to length n with zeros, or truncate.
        h_.assign(n_, 0.0);
        const std::size_t h_len = static_cast<std::size_t>(h_input.n_elem);
        for (std::size_t i = 0; i < std::min(n_, h_len); ++i) {
            h_[i] = h_input[i];
        }

        // Cardinalities: uniform K across all nodes.
        arma::uvec cards(n_);
        for (std::size_t i = 0; i < n_; ++i) cards[i] = K_;

        // log-density closure: captures beta, h by value.
        const double beta_cap = beta_;
        const std::vector<double> h_cap = h_;
        auto lp_fn = [beta_cap, h_cap](const arma::uvec& z,
                                        const block_context&) -> double {
            double v = 0.0;
            for (std::size_t i = 0; i + 1 < z.n_elem; ++i) {
                if (z[i] == z[i + 1]) v += beta_cap;
            }
            for (std::size_t i = 0; i < z.n_elem; ++i) {
                if (z[i] == 1u) v += h_cap[i];
            }
            return v;
        };

        // ---- VI block config ------------------------------------------------
        mean_field_categorical_vi_block_config cfg;
        cfg.name              = "z";
        cfg.cardinalities     = cards;
        cfg.log_density       = lp_fn;
        cfg.exact_enumeration = exact_enumeration;
        cfg.exact_state_cap   = 4096;
        cfg.n_mc_samples      = 32;
        cfg.optimizer.gamma_0              = 0.1;
        cfg.optimizer.rho                  = 0.5;
        cfg.optimizer.tau                  = 0.01;
        cfg.optimizer.inner_iter_per_epoch = 200;
        cfg.optimizer.max_epochs           = 25;
        cfg.optimizer.S_khat               = 1000;
        cfg.init_random_eps                = 0.1;
        cfg.init_rng_seed                  = static_cast<std::uint64_t>(rng_seed);

        // ---- shared_data setup ---------------------------------------------
        impl_->data().set("beta", arma::vec{beta_});
        arma::vec h_arma(n_);
        for (std::size_t i = 0; i < n_; ++i) h_arma[i] = h_[i];
        impl_->data().set("h", h_arma);

        // Initial z sample (uniform random).
        arma::vec z_init(n_);
        std::uniform_int_distribution<std::size_t> Ud(0, K_ - 1);
        for (std::size_t i = 0; i < n_; ++i) {
            z_init[i] = static_cast<double>(Ud(rng_));
        }
        impl_->data().set("z", z_init);

        // Gibbs DAG: z reads beta and h.
        impl_->data().declare_dependencies("z", {"beta", "h"});

        // Generative-DAG context edges (VIZ-ONLY): beta and h are
        // hyperparameters generating z.
        impl_->data().declare_context_edges("beta", {"z"});
        impl_->data().declare_context_edges("h",    {"z"});

        // No predict DAG / stochastic refresher: predict_at draws fresh
        // q-samples directly via current_sample (no graph traversal).

        impl_->add_child(std::make_unique<mean_field_categorical_vi_block>(
            std::move(cfg)));

        if (keep_history_) impl_->set_keep_history(true);
    }

    // ---- Canonical 6-method interface (backend-neutral) -----------------

    void step(int n_steps) {
        if (n_steps < 0) ai4b::stop("n_steps must be >= 0");
        for (int i = 0; i < n_steps; ++i) impl_->step(rng_);
    }

    AI4BayesCode::state_map get_current() const {
        // phi flattened COLUMN-MAJOR to a flat length-(n*K) arma::vec.
        const auto* vi = dynamic_cast<const mean_field_categorical_vi_block*>(
            &impl_->child(0));
        if (!vi) ai4b::stop("internal: child 0 is not mean_field_categorical_vi_block");
        const arma::vec phi_flat_block = vi->current();   // row-major n*K layout
        arma::vec phi_colmajor(n_ * K_);
        for (std::size_t i = 0; i < n_; ++i) {
            for (std::size_t k = 0; k < K_; ++k) {
                // column-major position of (i, k) in an n×K matrix is k*n + i
                phi_colmajor[k * n_ + i] = phi_flat_block[i * K_ + k];
            }
        }

        AI4BayesCode::state_map out;
        out["phi"]       = phi_colmajor;
        out["z"]         = impl_->data().get("z");
        out["beta"]      = impl_->data().get("beta");
        out["h"]         = impl_->data().get("h");
        out["elbo"]      = arma::vec{ vi->current_elbo() };
        out["converged"] = arma::vec{ vi->converged() ? 1.0 : 0.0 };
        out["epoch"]     = arma::vec{ static_cast<double>(vi->epoch()) };
        return out;
    }

    void set_current(const AI4BayesCode::state_map& params) {
        auto* vi = dynamic_cast<mean_field_categorical_vi_block*>(
            &impl_->child(0));
        if (!vi) ai4b::stop("internal: child 0 is not mean_field_categorical_vi_block");

        // phi arrives as the flat COLUMN-MAJOR length-(n*K) vector (matrices
        // are vectorised column-major under the backend-neutral convention).
        // The VI block expects ROW-MAJOR (i*K + k) layout, so transpose.
        auto it_phi = params.find("phi");
        if (it_phi != params.end()) {
            const arma::vec& phi_in = it_phi->second;
            if (phi_in.n_elem != n_ * K_) {
                ai4b::stop("set_current: phi must be a flat length n*K vector "
                           "(column-major n x K)");
            }
            arma::vec phi_block(n_ * K_);   // row-major layout for the block
            for (std::size_t i = 0; i < n_; ++i) {
                for (std::size_t k = 0; k < K_; ++k) {
                    phi_block[i * K_ + k] = phi_in[k * n_ + i];
                }
            }
            vi->set_current(phi_block);
        }

        auto it_z = params.find("z");
        if (it_z != params.end()) {
            const arma::vec& z_new = it_z->second;
            if (z_new.n_elem != n_) {
                ai4b::stop("set_current: z must have length n");
            }
            for (std::size_t i = 0; i < n_; ++i) {
                const double zi = z_new[i];
                if (zi < 0.0 || zi >= static_cast<double>(K_) ||
                    std::floor(zi) != zi) {
                    ai4b::stop("set_current: z entries must be integers in 0..K-1");
                }
            }
            impl_->data().set("z", z_new);
        }

        auto it_beta = params.find("beta");
        if (it_beta != params.end()) {
            const arma::vec& beta_new = it_beta->second;
            if (beta_new.n_elem != 1)
                ai4b::stop("set_current: beta must be length 1");
            if (!(beta_new[0] >= 0.0))
                ai4b::stop("set_current: beta must be >= 0");
            impl_->data().set("beta", beta_new);
            beta_ = beta_new[0];
            // NOTE: the log_density closure captured beta at construction.
            // Updating beta in shared_data is informational only; for
            // re-tuning the VI under a new beta, reconstruct the wrapper.
        }

        auto it_h = params.find("h");
        if (it_h != params.end()) {
            const arma::vec& h_new = it_h->second;
            if (h_new.n_elem != n_)
                ai4b::stop("set_current: h must have length n");
            impl_->data().set("h", h_new);
            for (std::size_t i = 0; i < n_; ++i) h_[i] = h_new[i];
            // Same caveat: log_density captured h at construction.
        }
    }

    AI4BayesCode::history_map predict_at(const AI4BayesCode::state_map& new_data) const {
        // Returns N fresh q-samples (default N=100). The VI block is a
        // deterministic q at the current eta; sampling from it is cheap.
        // Optional input key "n_draws" (length-1 vector) overrides N.
        std::size_t n_draws = 100;
        auto it_nd = new_data.find("n_draws");
        if (it_nd != new_data.end()) {
            if (it_nd->second.n_elem != 1)
                ai4b::stop("predict_at: n_draws must be length 1");
            const double Nd = it_nd->second[0];
            if (Nd < 0.0) ai4b::stop("predict_at: n_draws must be >= 0");
            n_draws = static_cast<std::size_t>(Nd);
        }
        const auto* vi = dynamic_cast<const mean_field_categorical_vi_block*>(
            &impl_->child(0));
        if (!vi) ai4b::stop("internal: child 0 is not mean_field_categorical_vi_block");

        arma::mat samples(n_draws, n_);   // (n_draws × n) integer-valued matrix
        for (std::size_t s = 0; s < n_draws; ++s) {
            const arma::vec z_s = vi->current_sample(predict_rng_);
            for (std::size_t i = 0; i < n_; ++i) {
                samples(s, i) = static_cast<double>(z_s[i]);
            }
        }
        AI4BayesCode::history_map out;
        out.emplace("z_samples", std::move(samples));
        return out;
    }

    AI4BayesCode::dag_info     get_dag()     const { return impl_->get_dag();     }
    AI4BayesCode::history_map  get_history() const { return impl_->get_history(); }

private:
    std::mt19937_64                  rng_;
    mutable std::mt19937_64          predict_rng_;
    std::unique_ptr<composite_block> impl_;
    bool                             keep_history_ = false;
    std::size_t                      n_  = 0;
    std::size_t                      K_  = 0;
    double                           beta_ = 0.0;
    std::vector<double>              h_;
};

// ============================================================================
//  Rcpp module
// ============================================================================

#ifdef AI4BAYESCODE_RCPP_MODULE
RCPP_MODULE(CategoricalIsingChainVI_module) {
    Rcpp::class_<CategoricalIsingChainVI>("CategoricalIsingChainVI")
        .constructor<int, int, double, arma::vec, bool, int>(
            "Legacy constructor; keep_history defaults to FALSE. Args: "
            "n_nodes, K, beta, h (length n_nodes, numeric vector), "
            "exact_enumeration, rng_seed.")
        .constructor<int, int, double, arma::vec, bool, int, bool>(
            "Construct with n_nodes (chain length, >= 2), K (per-node "
            "state count, >= 2), beta (coupling, >= 0), h (per-node "
            "external field, length n_nodes; 0 = symmetric chain), "
            "exact_enumeration (use full enumeration for gradient if "
            "K^n_nodes <= 4096; otherwise MC), rng_seed, keep_history. "
            "Discrete Potts chain mean-field VI via Bishop §10.1 + "
            "RAABBVI. Cross-validation at tests/test_categorical_"
            "meanfield_vi_chain.cpp.")
        .method("step",        &CategoricalIsingChainVI::step)
        .method("get_current", &CategoricalIsingChainVI::get_current,
                "Returns list(phi (flat n*K column-major vector; reshape "
                "matrix(phi, n, K)), z (latest q-sample), beta, h, elbo, "
                "converged, epoch).")
        .method("set_current", &CategoricalIsingChainVI::set_current,
                "Overwrite phi (flat length n*K column-major vector), z "
                "(length n integer indices), beta (length 1), and/or h "
                "(length n). beta/h updates are informational only; "
                "log_density was captured at construction. Reconstruct "
                "to re-tune VI under a new beta or h.")
        .method("predict_at",  &CategoricalIsingChainVI::predict_at,
                "Draw N fresh q-samples. Pass list(n_draws=100) (default "
                "N=100). Returns list(z_samples = n_draws × n integer "
                "matrix).")
        .method("get_dag",     &CategoricalIsingChainVI::get_dag)
        .method("get_history", &CategoricalIsingChainVI::get_history);
}
#endif // AI4BAYESCODE_RCPP_MODULE

// ============================================================================
//  pybind11 module
// ============================================================================

#ifdef AI4BAYESCODE_PYBIND_MODULE
#include "AI4BayesCode/pybind_casters.hpp"
PYBIND11_MODULE(CategoricalIsingChainVI, m) {
    AI4BayesCode::register_ai4bayescode_types(m);
    pybind11::class_<CategoricalIsingChainVI>(m, "CategoricalIsingChainVI")
        .def(pybind11::init<int, int, double, arma::vec, bool, int, bool>(),
             pybind11::arg("n_nodes"),
             pybind11::arg("K"),
             pybind11::arg("beta"),
             pybind11::arg("h"),
             pybind11::arg("exact_enumeration") = true,
             pybind11::arg("rng_seed") = 1,
             pybind11::arg("keep_history") = false)
        .def("step",        &CategoricalIsingChainVI::step, pybind11::arg("n_steps"))
        .def("get_current", &CategoricalIsingChainVI::get_current)
        .def("set_current", &CategoricalIsingChainVI::set_current, pybind11::arg("params"))
        .def("predict_at",  &CategoricalIsingChainVI::predict_at,  pybind11::arg("new_data"))
        .def("get_dag",     &CategoricalIsingChainVI::get_dag)
        .def("get_history", &CategoricalIsingChainVI::get_history);
}
#endif // AI4BAYESCODE_PYBIND_MODULE

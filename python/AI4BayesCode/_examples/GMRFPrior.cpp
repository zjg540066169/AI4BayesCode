// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
//  GMRFPrior.cpp
//
//  Pure-prior 2D ICAR sampler — first specialised-block demo for the
//  Block 2 v1.2 ship. Wires gmrf_precision_block (Rue 2001 sparse-
//  Cholesky direct sampling) through composite_block + a DUAL frontend
//  (RCPP_MODULE for R, PYBIND11_MODULE for Python).
//
//  Model
//  -----
//      x ~ N(0, (κ R)^{-1}),   with sum(x) = 0     (ICAR / IGMRF)
//
//  where R is the (improper) graph Laplacian of an L_x × L_y rectangular
//  lattice (4-NN or 8-NN, open or periodic). Specifically:
//      R[i, i] = deg(i)              (degree of vertex i in the graph)
//      R[i, j] = -1   if i ~ j       (adjacent in the lattice)
//      R[i, j] =  0   otherwise
//  R has a 1-dim null space spanned by the constant vector 1, hence
//  "intrinsic" GMRF (Rue & Held 2005 §3.3.2). Identifiability is
//  restored by the sum-to-zero constraint, enforced inside
//  gmrf_precision_block via Rue 2001 §3.1.3 (simplified single-
//  constraint case).
//
//  Parameters
//  ----------
//      x        latent field (length L_x * L_y) — sampled by
//               gmrf_precision_block
//      kappa    precision scale (scalar, > 0) — fixed at construction
//               by default; can be overwritten via set_current("kappa")
//
//  No observation likelihood — pure prior demo. predict_at returns an
//  empty history_map. Documented exception per system_design.md §5.
//
//  JUSTIFICATION (Check #16): Fixed-dim continuous Gaussian with sparse
//  precision (system_design.md §11.1 class 1, specialised efficiency
//  path). gmrf_precision_block is the library-blessed sparse-Cholesky
//  direct sampler (Rue 2001 §2 + §3.1.2 + §3.1.3 simplified). Check
//  #15 parity tests:
//    tests/test_gmrf_precision_block.cpp — 5 sub-tests covering
//      diagonal Q sanity, AR(1) Cov vs dense inverse, b ≠ 0 mean shift,
//      IGMRF sum-to-zero (exact constraint + projected-Cov match),
//      two-init R-hat across 4 chains on n=50.
// ============================================================================

// @example:R
//   library(AI4BayesCode)
//   # Pure-prior 2D ICAR: x ~ N(0, (kappa R)^{-1}) with sum(x)=0 on a
//   # 4x4 lattice. No data — x is drawn directly (Rue 2001 sparse-Cholesky),
//   # so each step is an exact i.i.d. prior draw (no observation likelihood).
//   m <- new(GMRFPrior,
//            4L,     # L_x : lattice width
//            4L,     # L_y : lattice height (N = L_x*L_y = 16 latent nodes)
//            2.0,    # kappa : precision scale (> 0), fixed at construction
//            FALSE,  # periodic : open boundary (no edge wrap)
//            FALSE,  # eight_nn : 4-nearest-neighbour adjacency
//            7L,     # rng_seed
//            TRUE)   # keep_history
//   m$step(2500); str(m$get_current())  # x is mean ~0, sum(x) ~ 0 (constraint)
// @example:python
//   import numpy as np, AI4BayesCode
//   # Pure-prior 2D ICAR: x ~ N(0, (kappa R)^{-1}) with sum(x)=0 on a 4x4
//   # lattice. No data — x is an exact sparse-Cholesky prior draw each step.
//   Mod = AI4BayesCode.source("GMRFPrior.cpp")
//   m = Mod.GMRFPrior(4, 4, 2.0, False, False, 7, True)
//   #             (L_x, L_y, kappa, periodic, eight_nn, rng_seed, keep_history)
//   m.step(2500)
//   cur = m.get_current()
//   x = np.asarray(cur["x"]).ravel()
//   print("mean(x)=", x.mean(), " sum(x)=", x.sum())  # ~0, ~0 (sum-to-zero)
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
#include "AI4BayesCode/gmrf_precision_block.hpp"
#include "AI4BayesCode/ising_cluster_block.hpp"  // reuse make_2d_lattice_edges

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>

using AI4BayesCode::block_context;
using AI4BayesCode::composite_block;
using AI4BayesCode::gmrf_precision_block;
using AI4BayesCode::gmrf_precision_block_config;
using AI4BayesCode::make_2d_lattice_edges;

// ----------------------------------------------------------------------------
//  Build the (improper) graph Laplacian R = D - W from an undirected
//  edge list. R is symmetric, sparse, with R[i,i] = degree(i),
//  R[i,j] = -1 for i ~ j, 0 otherwise. R has a 1-dim null space (the
//  constant vector); samples must be sum-to-zero-projected.
// ----------------------------------------------------------------------------
static arma::sp_mat
build_graph_laplacian(std::size_t n, const arma::umat& edges) {
    arma::sp_mat L(n, n);
    arma::vec    deg(n, arma::fill::zeros);
    for (std::size_t e = 0; e < edges.n_cols; ++e) {
        const std::size_t i = static_cast<std::size_t>(edges(0, e));
        const std::size_t j = static_cast<std::size_t>(edges(1, e));
        L(i, j) = -1.0;
        L(j, i) = -1.0;
        deg[i] += 1.0;
        deg[j] += 1.0;
    }
    for (std::size_t k = 0; k < n; ++k) L(k, k) = deg[k];
    return L;
}

class GMRFPrior {
public:
    GMRFPrior(int L_x, int L_y, double kappa,
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
          impl_(std::make_unique<composite_block>("GMRFPrior")),
          keep_history_(keep_history),
          L_x_(L_x), L_y_(L_y)
    {
        if (L_x_ < 1)        ai4b::stop("L_x must be >= 1");
        if (L_y_ < 1)        ai4b::stop("L_y must be >= 1");
        if (!(kappa > 0.0))  ai4b::stop("kappa must be > 0");

        N_ = static_cast<std::size_t>(L_x_) *
              static_cast<std::size_t>(L_y_);

        // ---- shared_data setup ----------------------------------------------
        impl_->data().set("kappa", arma::vec{kappa});
        impl_->data().set("x",     arma::vec(N_, arma::fill::zeros));

        // Gibbs DAG: x reads kappa each sweep.
        impl_->data().declare_dependencies("x", {"kappa"});

        // Generative-DAG context edge (VIZ-ONLY).
        impl_->data().declare_context_edges("kappa", {"x"});

        // No observation model / predict DAG.

        // ---- Build the graph Laplacian once (sparsity pattern fixed) -------
        arma::umat edges = make_2d_lattice_edges(
            static_cast<std::size_t>(L_x_),
            static_cast<std::size_t>(L_y_),
            periodic, eight_nn);
        arma::sp_mat R = build_graph_laplacian(N_, edges);

        // ---- gmrf_precision_block child ------------------------------------
        gmrf_precision_block_config cfg;
        cfg.name        = "x";
        cfg.n           = N_;
        cfg.Q_fn        = [R](const block_context& ctx) -> arma::sp_mat {
            const double k = ctx.at("kappa")[0];
            return k * R;
        };
        // b_fn left unset — pure prior, zero mean.
        cfg.sum_to_zero = true;
        // ridge_epsilon auto-set to 1e-8 in block constructor.
        impl_->add_child(std::make_unique<gmrf_precision_block>(std::move(cfg)));

        if (keep_history_) impl_->set_keep_history(true);
    }

    // ---- Canonical backend-neutral interface ----------------------------

    void step(int n_steps) {
        if (n_steps < 0) ai4b::stop("n_steps must be >= 0");
        for (int i = 0; i < n_steps; ++i) impl_->step(rng_);
    }

    AI4BayesCode::state_map get_current() const {
        AI4BayesCode::state_map out;
        out["x"]     = impl_->data().get("x");      // length N flat arma::vec
        out["kappa"] = impl_->data().get("kappa");  // length-1 arma::vec
        return out;
    }

    void set_current(const AI4BayesCode::state_map& params) {
        auto it_x = params.find("x");
        if (it_x != params.end()) {
            auto* x_blk = dynamic_cast<gmrf_precision_block*>(
                &impl_->child(0));
            if (!x_blk) {
                ai4b::stop("internal: child 0 is not gmrf_precision_block");
            }
            const arma::vec& x_new = it_x->second;
            if (x_new.n_elem != N_) {
                ai4b::stop("set_current: x length must equal L_x * L_y");
            }
            x_blk->set_current(x_new);
            impl_->data().set("x", x_new);
        }
        auto it_kappa = params.find("kappa");
        if (it_kappa != params.end()) {
            const arma::vec& kappa_new = it_kappa->second;
            if (kappa_new.n_elem != 1) {
                ai4b::stop("set_current: kappa must be a length-1 vector");
            }
            if (!(kappa_new[0] > 0.0)) {
                ai4b::stop("set_current: kappa must be > 0");
            }
            impl_->data().set("kappa", kappa_new);
        }
    }

    // GMRFPrior has no observation model — no y_rep, no data inputs.
    // predict_at always returns an empty history_map (documented exception
    // per system_design.md §5). Non-empty input is rejected.
    AI4BayesCode::history_map predict_at(
            const AI4BayesCode::state_map& new_data) const {
        if (!new_data.empty()) {
            ai4b::stop("GMRFPrior::predict_at: no valid keys to replace "
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
    int                              L_x_ = 0, L_y_ = 0;
    std::size_t                      N_   = 0;
};

#ifdef AI4BAYESCODE_RCPP_MODULE
RCPP_MODULE(GMRFPrior_module) {
    Rcpp::class_<GMRFPrior>("GMRFPrior")
        .constructor<int, int, double, bool, bool, int>(
            "Legacy constructor; keep_history defaults to FALSE. Args: "
            "L_x, L_y, kappa, periodic, eight_nn, rng_seed.")
        .constructor<int, int, double, bool, bool, int, bool>(
            "Construct with L_x, L_y (lattice dims), kappa (precision "
            "scale, > 0), periodic (bool, wrap edges), eight_nn (bool, "
            "include diagonals), rng_seed, keep_history. Pure-prior 2D "
            "ICAR via gmrf_precision_block sparse-Cholesky direct "
            "sampling (Rue 2001). Sum-to-zero constraint enforced. "
            "Check #15 parity tests under tests/test_gmrf_precision_block.cpp.")
        .method("step",        &GMRFPrior::step)
        .method("get_current", &GMRFPrior::get_current)
        .method("set_current", &GMRFPrior::set_current,
                "Overwrite x (length L_x*L_y) and/or kappa (length-1, > 0) "
                "from a named list. Unknown keys are silently ignored.")
        .method("predict_at",  &GMRFPrior::predict_at,
                "Returns empty list — GMRFPrior has no observation model. "
                "Pass an empty list; non-empty input is rejected.")
        .method("get_dag",     &GMRFPrior::get_dag)
        .method("get_history", &GMRFPrior::get_history);
}
#endif

#ifdef AI4BAYESCODE_PYBIND_MODULE
#include "AI4BayesCode/pybind_casters.hpp"
PYBIND11_MODULE(GMRFPrior, m) {
    AI4BayesCode::register_ai4bayescode_types(m);
    pybind11::class_<GMRFPrior>(m, "GMRFPrior")
        .def(pybind11::init<int, int, double, bool, bool, int, bool>(),
             pybind11::arg("L_x"),
             pybind11::arg("L_y"),
             pybind11::arg("kappa"),
             pybind11::arg("periodic") = false,
             pybind11::arg("eight_nn") = false,
             pybind11::arg("rng_seed") = 1,
             pybind11::arg("keep_history") = false)
        .def("step",        &GMRFPrior::step, pybind11::arg("n_steps"))
        .def("get_current", &GMRFPrior::get_current)
        .def("set_current", &GMRFPrior::set_current, pybind11::arg("params"))
        .def("predict_at",  &GMRFPrior::predict_at,  pybind11::arg("new_data"))
        .def("get_dag",     &GMRFPrior::get_dag)
        .def("get_history", &GMRFPrior::get_history);
}
#endif

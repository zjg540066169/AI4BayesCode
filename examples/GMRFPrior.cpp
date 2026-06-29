// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
//  GMRFPrior.cpp
//
//  Pure-prior 2D ICAR sampler — first specialised-block demo for the
//  Block 2 v1.2 ship. Wires gmrf_precision_block (Rue 2001 sparse-
//  Cholesky direct sampling) through composite_block.
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
//  No observation likelihood — pure prior demo.
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

// Frontend-INDEPENDENT example: pure standalone C++ (NO Rcpp, NO pybind).
// Compile + run directly: it has an int main() that samples from the prior
// and checks its moments against the known ICAR covariance. No R / Python
// binding is built or required.

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
#include "AI4BayesCode/gmrf_precision_block.hpp"
#include "AI4BayesCode/ising_cluster_block.hpp"  // reuse make_2d_lattice_edges

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>

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
          impl_(std::make_unique<composite_block>("GMRFPrior")),
          keep_history_(keep_history),
          L_x_(L_x), L_y_(L_y)
    {
        if (L_x_ < 1)        throw std::runtime_error("L_x must be >= 1");
        if (L_y_ < 1)        throw std::runtime_error("L_y must be >= 1");
        if (!(kappa > 0.0))  throw std::runtime_error("kappa must be > 0");

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

    // ---- Canonical neutral-typed interface ----------------------------

    void step(int n_steps) {
        if (n_steps < 0) throw std::runtime_error("n_steps must be >= 0");
        for (int i = 0; i < n_steps; ++i) impl_->step(rng_);
    }

    AI4BayesCode::state_map get_current() const {
        AI4BayesCode::state_map out;
        out["x"]     = impl_->data().get("x");
        out["kappa"] = impl_->data().get("kappa");
        return out;
    }

    void set_current(const AI4BayesCode::state_map& params) {
        auto it_x = params.find("x");
        if (it_x != params.end()) {
            auto* x_blk = dynamic_cast<gmrf_precision_block*>(
                &impl_->child(0));
            if (!x_blk) {
                throw std::runtime_error(
                    "internal: child 0 is not gmrf_precision_block");
            }
            const arma::vec& x_new = it_x->second;
            if (x_new.n_elem != N_) {
                throw std::runtime_error(
                    "set_current: x length must equal L_x * L_y");
            }
            x_blk->set_current(x_new);
            impl_->data().set("x", x_new);
        }
        auto it_k = params.find("kappa");
        if (it_k != params.end()) {
            const arma::vec& kappa_new = it_k->second;
            if (kappa_new.n_elem != 1) {
                throw std::runtime_error(
                    "set_current: kappa must be a length-1 vector");
            }
            if (!(kappa_new[0] > 0.0)) {
                throw std::runtime_error("set_current: kappa must be > 0");
            }
            impl_->data().set("kappa", kappa_new);
        }
    }

    AI4BayesCode::dag_info     get_dag()     const { return impl_->get_dag();     }
    AI4BayesCode::history_map  get_history() const { return impl_->get_history(); }

    std::size_t dim() const noexcept { return N_; }

private:
    std::mt19937_64                  rng_;
    std::unique_ptr<composite_block> impl_;
    bool                             keep_history_ = false;
    int                              L_x_ = 0, L_y_ = 0;
    std::size_t                      N_   = 0;
};

//==============================================================================
//  Standalone FRONTEND-INDEPENDENT demo (pure C++; no Rcpp, no pybind).
//
//  GMRFPrior is a PURE-PRIOR sampler — there is no observation likelihood, so
//  "recovery" means: the draws must reproduce the prior's KNOWN moments.
//
//  For the ICAR prior  x ~ N(0, (κ R)^{-1})  with the sum-to-zero constraint,
//  the constrained draws are supported on the orthogonal complement of the
//  constant vector 1, and their covariance equals the scaled Moore-Penrose
//  pseudo-inverse of the (rank-deficient) graph Laplacian:
//
//        Cov(x) = (1/κ) · R^+ ,
//
//  where R^+ is the pseudo-inverse of R restricted to range(R) = 1^⊥
//  (Rue & Held 2005 §3.3.2 / §3.4.2). We check three things against this
//  closed form:
//    (1) every draw is exactly sum-to-zero (constraint honoured);
//    (2) the empirical mean is ≈ 0;
//    (3) the empirical covariance ≈ (1/κ) R^+ in relative Frobenius norm.
//==============================================================================
#include <cstdio>
int main() {
    // ---- small lattice so we can form R^+ densely -------------------------
    const int    L_x   = 4;
    const int    L_y   = 4;
    const double kappa = 2.0;
    const std::size_t N = static_cast<std::size_t>(L_x) *
                          static_cast<std::size_t>(L_y);   // 16 nodes

    GMRFPrior model(L_x, L_y, kappa,
                    /*periodic=*/false, /*eight_nn=*/false,
                    /*rng_seed=*/123, /*keep_history=*/false);

    // ---- closed-form target covariance:  Cov(x) = (1/κ) R^+ ---------------
    // Rebuild R densely here (same lattice/edges as the model) and form its
    // pseudo-inverse. R^+ already lives on 1^⊥, matching the constrained draw.
    arma::umat edges = make_2d_lattice_edges(
        static_cast<std::size_t>(L_x), static_cast<std::size_t>(L_y),
        /*periodic=*/false, /*eight_nn=*/false);
    arma::sp_mat R_sp = build_graph_laplacian(N, edges);
    arma::mat    R    = arma::mat(R_sp);
    arma::mat    Rplus = arma::pinv(R);              // R^+ on range(R) = 1^⊥
    arma::mat    Cov_true = Rplus / kappa;

    // ---- warmup (direct sampler: not strictly needed, but harmless) -------
    model.step(50);

    // ---- collect draws ----------------------------------------------------
    const int    M = 200000;
    arma::vec    mean_acc(N, arma::fill::zeros);
    arma::mat    cov_acc(N, N, arma::fill::zeros);
    double       max_abs_sum = 0.0;   // worst per-draw |sum(x)| -> constraint

    for (int s = 0; s < M; ++s) {
        model.step(1);
        const arma::vec x = model.get_current().at("x");
        max_abs_sum = std::max(max_abs_sum, std::abs(arma::accu(x)));
        mean_acc += x;
        cov_acc  += x * x.t();
    }
    mean_acc /= static_cast<double>(M);
    cov_acc  /= static_cast<double>(M);
    // E[x]≈0 so the empirical second moment ≈ covariance; subtract mean outer
    // product for the unbiased-ish estimate.
    arma::mat Cov_emp = cov_acc - mean_acc * mean_acc.t();

    // ---- compare ----------------------------------------------------------
    const double mean_max   = arma::abs(mean_acc).max();
    const double cov_relerr =
        arma::norm(Cov_emp - Cov_true, "fro") /
        arma::norm(Cov_true, "fro");

    std::printf("GMRFPrior demo: ICAR on %dx%d lattice (N=%zu), kappa=%.1f, "
                "M=%d draws\n", L_x, L_y, N, kappa, M);
    std::printf("  max |sum(x)| over all draws = %.3e   (constraint, want ~0)\n",
                max_abs_sum);
    std::printf("  max |empirical mean|        = %.4f   (want ~0)\n", mean_max);
    std::printf("  Cov relative Frobenius err  = %.4f   (emp vs (1/kappa) R^+)\n",
                cov_relerr);

    // Diagonal spot-check (a couple of marginal variances).
    std::printf("  diag var: node0 emp=%.4f true=%.4f | node5 emp=%.4f "
                "true=%.4f\n",
                Cov_emp(0, 0), Cov_true(0, 0),
                Cov_emp(5, 5), Cov_true(5, 5));

    const bool ok =
        (max_abs_sum < 1e-8) &&     // sum-to-zero is exact per draw
        (mean_max    < 0.02) &&     // Monte-Carlo mean ~ 0
        (cov_relerr  < 0.03);       // covariance matches (1/kappa) R^+

    std::printf("%s\n", ok
        ? "[demo PASS] ICAR draws match prior moments (1/kappa) R^+"
        : "[demo FAIL] sampled moments do not match the ICAR prior");
    return ok ? 0 : 1;
}

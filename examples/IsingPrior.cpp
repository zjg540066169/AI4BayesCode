// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
//  IsingPrior.cpp
//
//  Pure-prior 2D Ising / Potts sampler — first specialized-block demo for
//  AI4BayesCode v1.2. Wires ising_cluster_block (Swendsen-Wang cluster
//  sweep) through composite_block. (Frontend-independent standalone build:
//  no Rcpp / pybind binding — see int main() at the bottom.)
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

// Frontend-INDEPENDENT example: pure standalone C++ (NO Rcpp, NO pybind).
// Compile + run directly: it has an int main() that draws from the Ising /
// Potts PRIOR and checks the draws against known closed-form prior moments.
// No R / Python binding is built or required.

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
#include "AI4BayesCode/ising_cluster_block.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>

using AI4BayesCode::block_context;
using AI4BayesCode::composite_block;
using AI4BayesCode::ising_cluster_block;
using AI4BayesCode::ising_cluster_block_config;
using AI4BayesCode::make_2d_lattice_edges;

class IsingPrior {
public:
    IsingPrior(int L_x, int L_y, int Q, double beta,
                bool periodic, bool eight_nn,
                int rng_seed,
                bool keep_history = false)
        : rng_(rng_seed == 0
                   ? std::mt19937_64{std::random_device{}()}
                   : std::mt19937_64{static_cast<std::uint64_t>(rng_seed)}),
          impl_(std::make_unique<composite_block>("IsingPrior")),
          keep_history_(keep_history),
          L_x_(L_x), L_y_(L_y), Q_(Q)
    {
        if (L_x_ < 1)         throw std::runtime_error("L_x must be >= 1");
        if (L_y_ < 1)         throw std::runtime_error("L_y must be >= 1");
        if (Q_   < 2)         throw std::runtime_error("Q must be >= 2");
        if (!(beta >= 0.0))   throw std::runtime_error("beta must be >= 0");

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

    void step(int n_steps) {
        if (n_steps < 0) throw std::runtime_error("n_steps must be >= 0");
        for (int i = 0; i < n_steps; ++i) impl_->step(rng_);
    }

    // Current latent state x (length L_x*L_y, entries in {0..Q-1}).
    const arma::vec& current_x() const { return impl_->data().get("x"); }

    std::size_t n_sites() const noexcept { return N_; }

private:
    std::mt19937_64                  rng_;
    std::unique_ptr<composite_block> impl_;
    bool                             keep_history_ = false;
    int                              L_x_ = 0, L_y_ = 0, Q_ = 0;
    std::size_t                      N_   = 0;
};

//==============================================================================
//  Standalone FRONTEND-INDEPENDENT demo (pure C++; no Rcpp, no pybind).
//
//  IsingPrior is a PRIOR-ONLY sampler (no likelihood), so we validate it
//  against closed-form prior moments instead of "recovering" parameters:
//
//   (A) Label symmetry (holds at ANY β): the Ising prior
//         pi(x) ∝ exp{ β Σ_{i~j} I[x_i = x_j] }
//       is invariant under the global swap 0<->1, so each site marginal is
//       P(x_i = 0) = P(x_i = 1) = 1/2.  Hence E[x_i] = 0.5 and, averaged over
//       sites and sweeps, the empirical site-state mean must be ≈ 0.5.  We
//       run a ferromagnetic β = 0.44 (near the 2D Ising critical point) where
//       single-site Gibbs would mix slowly — the SW cluster move flips whole
//       domains, so symmetry is realised in a modest number of sweeps.
//
//   (B) β = 0 independence: p_bond = 1 - exp(0) = 0, so every site is
//       recoloured iid Uniform{0..Q-1}.  For Ising (Q=2) the per-sweep
//       magnetisation fraction (mean of x) is then Binomial(N, 1/2)/N with
//       Var = 1/(4N); its sample variance across sweeps must match 1/(4N).
//==============================================================================
#include <cstdio>
int main() {
    const int    L      = 16;            // 16 x 16 lattice (N = 256 sites)
    const int    Q      = 2;             // Ising
    const double beta_c = 0.44;          // near 2D Ising critical coupling
    const std::size_t N = static_cast<std::size_t>(L) * L;

    // ---- (A) label-symmetry check at ferromagnetic beta --------------------
    IsingPrior model(L, L, Q, beta_c,
                     /*periodic=*/true, /*eight_nn=*/false,
                     /*rng_seed=*/12345);
    model.step(200);                     // burn-in cluster sweeps

    const int    n_keep = 4000;
    double       sum_state = 0.0;        // accumulates mean-over-sites per sweep
    long long    count = 0;
    for (int s = 0; s < n_keep; ++s) {
        model.step(1);
        const arma::vec& x = model.current_x();
        for (std::size_t i = 0; i < N; ++i) sum_state += x[i];
        count += static_cast<long long>(N);
    }
    const double mean_state = sum_state / static_cast<double>(count);
    // MC SE of a mean of ~0.5 Bernoulli's over n_keep nearly-independent SW
    // sweeps: with N=256 sites the per-sweep mean has SE ~ 0.5/sqrt(256), and
    // SW decorrelates fast, so a tolerance of 0.03 is comfortably honest.
    const double tol_sym = 0.03;
    const bool ok_sym = std::abs(mean_state - 0.5) < tol_sym;

    // ---- (B) beta = 0 independence: per-sweep magnetisation variance -------
    IsingPrior model0(L, L, Q, /*beta=*/0.0,
                      /*periodic=*/true, /*eight_nn=*/false,
                      /*rng_seed=*/999);
    model0.step(20);                     // warm a few iid recolourings
    const int n0 = 4000;
    double sm = 0.0, sm2 = 0.0;
    for (int s = 0; s < n0; ++s) {
        model0.step(1);
        const arma::vec& x = model0.current_x();
        double mag = 0.0;
        for (std::size_t i = 0; i < N; ++i) mag += x[i];
        mag /= static_cast<double>(N);   // fraction of "1" sites this sweep
        sm  += mag;
        sm2 += mag * mag;
    }
    const double m_bar   = sm / n0;
    const double m_var   = sm2 / n0 - m_bar * m_bar;   // empirical Var(mag)
    const double var_th  = 0.25 / static_cast<double>(N);  // 1/(4N)
    // sample variance of a variance is noisy; accept within 25% relative.
    const bool ok_indep =
        std::abs(m_bar - 0.5) < 0.02 &&
        std::abs(m_var - var_th) < 0.25 * var_th;

    std::printf("IsingPrior demo (L=%d, Q=%d):\n", L, Q);
    std::printf("  (A) beta=%.2f  site-state mean = %.4f (prior 0.5000, "
                "|diff|=%.4f, tol %.2f)  -> %s\n",
                beta_c, mean_state, std::abs(mean_state - 0.5), tol_sym,
                ok_sym ? "ok" : "FAIL");
    std::printf("  (B) beta=0.00  mag mean = %.4f, mag var = %.6f "
                "(theory 1/(4N)=%.6f)  -> %s\n",
                m_bar, m_var, var_th, ok_indep ? "ok" : "FAIL");

    const bool ok = ok_sym && ok_indep;
    std::printf("%s\n", ok
        ? "[demo PASS] IsingPrior draws match closed-form prior moments"
        : "[demo FAIL]");
    return ok ? 0 : 1;
}

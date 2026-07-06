// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
//  IsingHiddenPotts.cpp
//
//  Hidden-Potts image segmentation demo — exercises the v1.2.1
//  ising_cluster_block EXTERNAL FIELD + PARTIAL DECOUPLING path.
//  (Frontend-independent standalone build: no Rcpp / pybind — see int main().)
//
//  Model (hidden Potts / hidden MRF; Higdon 1998, Moores et al.)
//  ------------------------------------------------------------
//      z_i ∈ {0,1}      latent labels on an L×L lattice
//      spatial prior:   p(z) ∝ exp{ β Σ_{i~j} I[z_i = z_j] }
//      emission:        y_i | z_i = k  ~  N(μ_k, σ²)
//   => external field (log-potential, constant term drops in the recolour):
//          α_i(k) = log N(y_i; μ_k, σ²) = -0.5 (y_i - μ_k)² / σ²
//
//  The labels z are sampled by ising_cluster_block with the field supplied
//  via `field_key`: a sibling emission block would publish log p(y_i | θ_k)
//  to shared_data each sweep; here μ, σ are fixed so the field is published
//  once. Partial decoupling δ < 1 keeps the cluster move mixing well despite
//  the strong data field (Higdon §2.3.2) — with δ = 1 the SW clusters form
//  ignoring the field and mix poorly.
//
//  Validation (demo, not a unit test): with a well-separated, spatially
//  coherent truth the posterior label marginals recover the segmentation
//  with high accuracy, and two independent chains (init all-0 vs all-1)
//  agree (two-chain R-hat < 1.01 — the field breaks the 0↔1 symmetry, so
//  there is NO label switching to confound the diagnostic).
//
//  JUSTIFICATION (Check #16): discrete-MRF target with strong local
//  dependence + external field — system_design.md §11.2(b). SW cluster
//  moves with partial decoupling are the standard remedy (Higdon 1998).
//
//  Check #15 parity tests: tests/test_ising_cluster_block.cpp
//    (T6 external field vs enumeration, T7 per-edge β, T8 partial-
//     decoupling target-invariance; each gated on two-chain R-hat < 1.01).
// ============================================================================

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

class IsingHiddenPotts {
public:
    IsingHiddenPotts(int L, double beta, double delta,
                     double mu0, double mu1, double sigma,
                     int data_seed, int rng_seed, int init_label)
        : rng_(static_cast<std::uint64_t>(rng_seed)),
          impl_(std::make_unique<composite_block>("IsingHiddenPotts")),
          L_(L)
    {
        if (L_ < 2)                          throw std::runtime_error("L must be >= 2");
        if (!(beta >= 0.0))                  throw std::runtime_error("beta must be >= 0");
        if (!(delta >= 0.0 && delta <= 1.0)) throw std::runtime_error("delta in [0,1]");
        if (!(sigma > 0.0))                  throw std::runtime_error("sigma must be > 0");
        N_ = static_cast<std::size_t>(L_) * static_cast<std::size_t>(L_);
        const std::size_t Q = 2;

        // ---- synthetic truth + data (left half = 0, right half = 1) --------
        z_true_.set_size(N_);
        for (int r = 0; r < L_; ++r)
            for (int c = 0; c < L_; ++c)
                z_true_[static_cast<std::size_t>(r) * L_ + c] =
                    (c >= L_ / 2) ? 1.0 : 0.0;

        std::mt19937_64 dgen(static_cast<std::uint64_t>(data_seed));
        std::normal_distribution<double> noise(0.0, sigma);
        arma::vec y(N_);
        for (std::size_t i = 0; i < N_; ++i)
            y[i] = ((z_true_[i] > 0.5) ? mu1 : mu0) + noise(dgen);

        // ---- external field α_i(k) = -0.5 (y_i - μ_k)² / σ² ----------------
        arma::mat alpha(N_, Q);
        const double inv2s2 = 0.5 / (sigma * sigma);
        for (std::size_t i = 0; i < N_; ++i) {
            alpha(i, 0) = -inv2s2 * (y[i] - mu0) * (y[i] - mu0);
            alpha(i, 1) = -inv2s2 * (y[i] - mu1) * (y[i] - mu1);
        }

        // ---- shared_data + Gibbs wiring ------------------------------------
        impl_->data().set("beta", arma::vec{beta});
        impl_->data().set("log_lik", arma::vectorise(alpha));  // n*Q, col-major
        arma::vec z_init(N_);
        z_init.fill(static_cast<double>(init_label));
        impl_->data().set("z", z_init);
        impl_->data().declare_dependencies("z", {"beta", "log_lik"});
        impl_->data().declare_context_edges("beta", {"z"});
        impl_->data().declare_context_edges("log_lik", {"z"});

        // ---- ising_cluster_block with external field + partial decoupling --
        ising_cluster_block_config cfg;
        cfg.name          = "z";
        cfg.n_vertices    = N_;
        cfg.n_states      = Q;
        cfg.edges         = make_2d_lattice_edges(
            static_cast<std::size_t>(L_), static_cast<std::size_t>(L_),
            /*periodic=*/false, /*eight_nn=*/false);
        cfg.beta_key      = "beta";
        cfg.beta_default  = beta;
        cfg.field_key     = "log_lik";   // <- sibling-published emission field
        cfg.delta_default = delta;       // <- partial decoupling for field mixing
        cfg.initial_state = z_init;
        impl_->add_child(std::make_unique<ising_cluster_block>(std::move(cfg)));
    }

    void step(int n) { for (int i = 0; i < n; ++i) impl_->step(rng_); }
    const arma::vec& current_z() const { return impl_->data().get("z"); }
    arma::vec        true_z()   const { return z_true_; }
    std::size_t      n_sites()  const noexcept { return N_; }

private:
    std::mt19937_64                  rng_;
    std::unique_ptr<composite_block> impl_;
    int                              L_ = 0;
    std::size_t                      N_ = 0;
    arma::vec                        z_true_;
};

//==============================================================================
//  Standalone frontend-independent demo.
//==============================================================================
#include <cstdio>
#include <vector>

int main() {
    const int    L     = 16;    // 16x16 lattice (N = 256)
    const double beta  = 0.7;   // spatial smoothing
    const double delta = 0.6;   // partial decoupling (δ<1) for field mixing
    const double mu0 = 0.0, mu1 = 2.0, sigma = 1.0;   // SNR = (μ1-μ0)/σ = 2

    auto run = [&](int init, int seed, arma::vec& p1,
                   std::vector<double>& up, arma::vec& ztrue) {
        IsingHiddenPotts m(L, beta, delta, mu0, mu1, sigma,
                           /*data_seed=*/2024, seed, init);
        ztrue = m.true_z();
        const std::size_t N = m.n_sites();
        m.step(300);                                   // burn-in
        const int M = 4000;
        p1.zeros(N); up.clear(); up.reserve(M);
        for (int s = 0; s < M; ++s) {
            m.step(1);
            const arma::vec& z = m.current_z();
            double u = 0.0;
            for (std::size_t i = 0; i < N; ++i) { p1[i] += z[i]; u += z[i]; }
            up.push_back(u);
        }
        p1 /= static_cast<double>(M);
    };

    arma::vec p1a, p1b, zt; std::vector<double> u1, u2;
    run(0, 101, p1a, u1, zt);
    run(1, 202, p1b, u2, zt);
    const arma::vec p1 = 0.5 * (p1a + p1b);

    // Segmentation accuracy (threshold posterior marginals at 0.5).
    const std::size_t N = zt.n_elem;
    int correct = 0;
    for (std::size_t i = 0; i < N; ++i)
        if (((p1[i] >= 0.5) ? 1 : 0) == static_cast<int>(zt[i] + 0.5)) ++correct;
    const double acc = static_cast<double>(correct) / static_cast<double>(N);

    // Two-chain Gelman-Rubin R-hat on the total up-count Σz.
    auto rhat2 = [](const std::vector<double>& a, const std::vector<double>& b) {
        const double M = static_cast<double>(a.size());
        auto mean = [](const std::vector<double>& v) {
            double s = 0; for (double x : v) s += x; return s / v.size(); };
        const double ma = mean(a), mb = mean(b), gm = 0.5 * (ma + mb);
        const double B = M * ((ma - gm) * (ma - gm) + (mb - gm) * (mb - gm));
        auto wv = [](const std::vector<double>& v, double m) {
            double s = 0; for (double x : v) s += (x - m) * (x - m);
            return s / (v.size() - 1); };
        const double W = 0.5 * (wv(a, ma) + wv(b, mb));
        if (W <= 0.0) return 1.0;
        return std::sqrt(((M - 1.0) / M) * W + B / M) / std::sqrt(W);
    };
    const double rh = rhat2(u1, u2);

    const bool ok = (acc > 0.85) && (rh < 1.01);
    std::printf("IsingHiddenPotts demo (L=%d, beta=%.2f, delta=%.2f, SNR=%.1f):\n",
                L, beta, delta, (mu1 - mu0) / sigma);
    std::printf("  segmentation accuracy = %.4f  (tol > 0.85)\n", acc);
    std::printf("  two-chain R-hat (Sum z) = %.5f  (tol < 1.01)\n", rh);
    std::printf("%s\n", ok
        ? "[demo PASS] hidden-Potts recovers the segmentation via field + partial decoupling"
        : "[demo FAIL]");
    return ok ? 0 : 1;
}

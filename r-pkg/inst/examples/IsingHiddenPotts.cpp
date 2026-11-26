// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v3.0 or later
// (GPL-3.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
//  IsingHiddenPotts.cpp
//
//  Hidden-Potts image segmentation demo -- exercises the
//  ising_cluster_block EXTERNAL FIELD + PARTIAL DECOUPLING path.
//  Tri-module build: RCPP_MODULE / PYBIND11_MODULE / standalone int main().
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
//  the strong data field (Higdon Sec.2.3.2) -- with δ = 1 the SW clusters form
//  ignoring the field and mix poorly.
//
//  Parameters
//  ----------
//      z       latent label vector (length L·L, entries in {0,1}) --
//              sampled by ising_cluster_block. THIS is the only sampled
//              state (get_current() returns {"z"}). The synthetic truth
//              z* is NOT a parameter and is not exposed in get_current().
//
//  Posterior predictive (predict_at)
//  ---------------------------------
//      y_rep_i | z  ~  N(μ_{z_i}, σ²),   i = 1..N,
//  the emission model at the N training sites, with μ = (μ0, μ1) and σ the
//  fixed constructor values (shared keys "mu", "sigma"); every value is drawn
//  with predict_rng_ by the stochastic refresher of "y_rep" (predict edges
//  z, mu, sigma -> y_rep). keep_history = TRUE: one row per stored draw d,
//  from the labels z_d of that draw (row d of get_history()$z). keep_history
//  = FALSE, or last_draw_only = TRUE: one row, from the current labels. The
//  model has no covariate inputs, so predict_at() takes an empty map/list.
//
//  Validation (demo, not a unit test): with a well-separated, spatially
//  coherent truth the posterior label marginals recover the segmentation
//  with high accuracy, and two independent chains (init all-0 vs all-1)
//  agree (two-chain R-hat < 1.01 -- the field breaks the 0↔1 symmetry, so
//  there is NO label switching to confound the diagnostic).
//
//  Sampling note: discrete-MRF target with strong local
//  dependence + external field -- system_design.md Sec.11.2(b). SW cluster
//  moves with partial decoupling are the standard remedy (Higdon 1998).
//
// ============================================================================

// @example:R
//   library(AI4BayesCode)
//   ai4bayescode_example("IsingHiddenPotts")
//   # Hidden-Potts image segmentation on a 16x16 lattice: latent labels z with
//   # a ferromagnetic spatial prior (beta=0.7) and a Gaussian emission field
//   # (mu0=0, mu1=2, sigma=1). Partial decoupling delta=0.6 keeps the SW
//   # cluster move mixing despite the strong data field. We sample z; the
//   # emission field breaks the 0<->1 symmetry so there is no label switching.
//   # ---- Parallel chains + convergence diagnosis (default) ----
//   run <- ai4bayescode_run_chains(
//       function(seed) new(IsingHiddenPotts, 16L, 0.7, 0.6, 0.0, 2.0, 1.0,
//                          2024L, seed, 0L, TRUE),
//       n_chains = 4, n_burn = 1000, n_keep = 2000)
//   print(ai4bayescode_rhat_summary(run))          # CROSS-chain R-hat / ESS
//   ai4bayescode_diagnose(run$histories[[1]])      # chain 1: summary + plots
//   yrep <- predict(run)$y_rep    # posterior predictive y_rep_i ~ N(mu_{z_i}, sigma^2):
//                                 # 8000 x 256, one row per kept draw of the 4 chains
//   ai4bayescode_close_models(run)                 # stop the worker processes
//   # ---- Advanced: stateful single-chain control ----
//   m <- new(IsingHiddenPotts,
//            16L,        # L: lattice side (N = 256 sites)
//            0.7,        # beta: spatial interaction strength (>= 0)
//            0.6,        # delta: partial decoupling in [0,1]
//            0.0, 2.0,   # mu0, mu1: class means
//            1.0,        # sigma: emission sd (> 0)
//            2024L,      # data_seed: synthetic-truth / noise seed
//            42L,        # rng_seed
//            0L)         # init_label: all-0 initial state
//   m$step(2000)
//   z <- m$get_current()$z            # length-256 recovered labels in {0,1}
//   matrix(z, 16, 16)                 # segmentation: left half 0, right half 1
//   y_rep <- m$predict_at(list())$y_rep   # 1 x 256, drawn from the current z
// @example:python
//   import numpy as np, AI4BayesCode
//   # Hidden-Potts image segmentation on a 16x16 lattice: latent labels z with
//   # a ferromagnetic spatial prior (beta=0.7) and a Gaussian emission field
//   # (mu0=0, mu1=2, sigma=1). Partial decoupling delta=0.6 keeps the SW
//   # cluster move mixing despite the strong data field. We sample z; the
//   # emission field breaks the 0<->1 symmetry so there is no label switching.
//   Mod = AI4BayesCode.example("IsingHiddenPotts")
//   # ---- Parallel chains + diagnosis (default) ----
//   chains = AI4BayesCode.run_chains(
//       lambda seed: Mod.IsingHiddenPotts(16, 0.7, 0.6, 0.0, 2.0, 1.0,
//                                         2024, seed, 0, True),
//       seeds=[101, 202, 303, 404], n_burn=1000, n_keep=2000, n_cores=1)
//   print(AI4BayesCode.rhat_summary(chains))   # CROSS-chain R-hat / ESS
//   AI4BayesCode.diagnose(chains[0]["hist"])   # chain 1: summary + plots
//   yrep = AI4BayesCode.predict(chains)["y_rep"]   # posterior predictive y_rep:
//                                                  # (8000, 256), one row per kept draw
//   # ---- Advanced: stateful single-chain control ----
//   m = Mod.IsingHiddenPotts(16,        # L: lattice side (N = 256 sites)
//                            0.7,       # beta: spatial interaction (>= 0)
//                            0.6,       # delta: partial decoupling in [0,1]
//                            0.0, 2.0,  # mu0, mu1: class means
//                            1.0,       # sigma: emission sd (> 0)
//                            2024,      # data_seed
//                            42,        # rng_seed
//                            0)         # init_label: all-0 initial state
//   m.step(2000)
//   z = np.asarray(m.get_current()["z"])   # length-256 recovered labels {0,1}
//   print(z.reshape(16, 16))               # left half 0, right half 1
//   y_rep = m.predict_at({})["y_rep"]      # (1, 256), drawn from the current z
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

class IsingHiddenPotts : public AI4BayesCode::kernel_control_mixin<IsingHiddenPotts> {
    friend class AI4BayesCode::kernel_control_mixin<IsingHiddenPotts>;
public:
    IsingHiddenPotts(int L, double beta, double delta,
                     double mu0, double mu1, double sigma,
                     int data_seed, int rng_seed = 1, int init_label = 0,
                     bool keep_history = false)
        : rng_(static_cast<std::uint64_t>(rng_seed)),
          predict_rng_(static_cast<std::uint64_t>(rng_seed)
                       ^ 0x9E3779B97F4A7C15ULL),
          impl_(std::make_unique<composite_block>("IsingHiddenPotts")),
          L_(L),
          keep_history_(keep_history)
    {
        if (L_ < 2)                          ai4b::stop("L must be >= 2");
        if (!(beta >= 0.0))                  ai4b::stop("beta must be >= 0");
        if (!(delta >= 0.0 && delta <= 1.0)) ai4b::stop("delta in [0,1]");
        if (!(sigma > 0.0))                  ai4b::stop("sigma must be > 0");
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

        // ---- posterior predictive y_rep_i ~ N(μ_{z_i}, σ²) ------------------
        // μ and σ are fixed; no block depends on "mu" or "sigma", so the
        // sweeps are unchanged. predict_at() draws y_rep given the z of its
        // scratch copy of the shared data: the current labels, or a stored
        // draw's labels.
        impl_->data().set("mu",    arma::vec{mu0, mu1});
        impl_->data().set("sigma", arma::vec{sigma});
        impl_->data().set("y_rep", arma::vec(N_, arma::fill::zeros));
        impl_->data().declare_predict_edges("z",     {"y_rep"});
        impl_->data().declare_predict_edges("mu",    {"y_rep"});
        impl_->data().declare_predict_edges("sigma", {"y_rep"});
        impl_->data().register_stochastic_refresher(
            "y_rep",
            [](const AI4BayesCode::shared_data_t& d, std::mt19937_64& rng) {
                const arma::vec& z  = d.get("z");
                const arma::vec& mu = d.get("mu");
                const double     s  = d.get("sigma")[0];
                std::normal_distribution<double> nd(0.0, 1.0);
                arma::vec y_rep(z.n_elem);
                for (std::size_t i = 0; i < z.n_elem; ++i)
                    y_rep[i] = ((z[i] > 0.5) ? mu[1] : mu[0]) + s * nd(rng);
                return y_rep;
            });

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

        // Without this the composite keeps only the current draw, and any
        // diagnostic computed from get_history() sees a single row -- sd,
        // R-hat and ESS all come back NA.
        if (keep_history) impl_->set_keep_history(true);
    }

    // ---- Canonical backend-neutral interface ---------------------------

    void step() { step(1); }              // no-arg convenience: one sweep
    void step(int n_steps) {
        if (n_steps < 0) ai4b::stop("n_steps must be >= 0");
        for (int i = 0; i < n_steps; ++i) impl_->step(rng_);
    }

    AI4BayesCode::state_map get_current() const {
        AI4BayesCode::state_map out;
        out["z"] = impl_->data().get("z");   // length-N flat vec in {0,1}
        return out;
    }

    void set_current(const AI4BayesCode::state_map& params) {
        auto it_z = params.find("z");
        if (it_z != params.end()) {
            auto* z_blk = dynamic_cast<ising_cluster_block*>(
                &impl_->child(0));
            if (!z_blk) {
                ai4b::stop("internal: child 0 is not ising_cluster_block");
            }
            const arma::vec& z_new = it_z->second;
            if (z_new.n_elem != N_) {
                ai4b::stop("set_current: z length must equal L * L");
            }
            for (std::size_t i = 0; i < N_; ++i) {
                // z + 0.5 truncated toward zero is 0 or 1.
                const double v = z_new[i] + 0.5;
                if (!(v > -1.0 && v < 2.0)) {
                    ai4b::stop("set_current: z entries must be in {0,1}");
                }
            }
            // The condition and exception of ising_cluster_block::set_current,
            // which writes each entry as it checks it: an entry it rejects
            // would leave it partly loaded.
            for (std::size_t i = 0; i < N_; ++i) {
                const double lab = std::round(z_new[i]);
                if (lab != 0.0 && lab != 1.0)
                    throw std::invalid_argument(
                        "ising_cluster_block 'z': x entries must be in "
                        "{0..n_states-1}");
            }
            z_blk->set_current(z_new);
            impl_->sync_child(0);   // shared z = the block's rounded labels
        }
    }

    /// Posterior predictive y_rep at the N training sites. One-argument form:
    /// predict_at(new_data, false).
    AI4BayesCode::history_map predict_at(
            const AI4BayesCode::state_map& new_data) const {
        return predict_at(new_data, /*last_draw_only=*/false);
    }

    /// keep_history = true and last_draw_only = false: y_rep with one row per
    /// stored draw d, y_rep[d, i] ~ N(μ_{z_d[i]}, σ²) with z_d the labels of
    /// draw d (row d of get_history()$z). Otherwise one row, drawn from the
    /// current labels. Every value is drawn with predict_rng_.
    AI4BayesCode::history_map predict_at(
            const AI4BayesCode::state_map& new_data,
            bool last_draw_only) const {
        if (!new_data.empty()) {
            ai4b::stop("IsingHiddenPotts::predict_at: the model has no covariate "
                       "inputs; pass an empty map/list (y_rep at the training "
                       "sites).");
        }
        const bool use_history = keep_history_ && !last_draw_only;
        AI4BayesCode::history_map out;

        if (!use_history) {
            const block_context result =
                impl_->predict_at(block_context{}, predict_rng_);
            for (const auto& kv : result)
                out.emplace(kv.first, arma::mat(kv.second.t()));
            return out;
        }

        // History mode: the labels of each stored draw replace the shared z
        // in predict_at's scratch copy; μ and σ are the fixed shared values.
        const AI4BayesCode::history_map hist = impl_->get_history();
        const arma::mat& z_hist = hist.at("z");          // n_draws x N
        const arma::uword n_draws = z_hist.n_rows;
        for (arma::uword d = 0; d < n_draws; ++d) {
            block_context replaced;
            replaced["z"] = z_hist.row(d).t();
            const block_context result = impl_->predict_at(replaced, predict_rng_);
            for (const auto& kv : result) {
                auto it = out.find(kv.first);
                if (it == out.end())
                    it = out.emplace(kv.first,
                                     arma::mat(n_draws, kv.second.n_elem)).first;
                it->second.row(d) = kv.second.t();
            }
        }
        return out;
    }

    AI4BayesCode::dag_info     get_dag()     const { return impl_->get_dag();     }
    AI4BayesCode::history_map  get_history() const { return impl_->get_history(); }

    // Non-contract accessor used only by the standalone demo (the synthetic
    // truth z* is NOT a sampled parameter, so it is intentionally absent from
    // get_current()).
    const arma::vec& true_z()  const { return z_true_; }
    std::size_t      n_sites() const noexcept { return N_; }

private:
    std::mt19937_64                  rng_;
    mutable std::mt19937_64          predict_rng_;
    std::unique_ptr<composite_block> impl_;
    int                              L_ = 0;
    std::size_t                      N_ = 0;
    arma::vec                        z_true_;
    bool                             keep_history_ = false;
};

#ifdef AI4BAYESCODE_RCPP_MODULE
RCPP_MODULE(IsingHiddenPotts_module) {
    Rcpp::class_<IsingHiddenPotts>("IsingHiddenPotts")
        .constructor<int, double, double, double, double, double, int, int, int,
                     bool>(
            "Hidden-Potts image segmentation, keeping the draw history "
            "(pass keep_history = TRUE for any multi-draw diagnostic).")
        .constructor<int, double, double, double, double, double, int, int, int>(
            "Hidden-Potts image segmentation. Args: L (lattice side, N=L*L), "
            "beta (spatial interaction, >= 0), delta (partial decoupling in "
            "[0,1]), mu0, mu1 (class means), sigma (emission sd, > 0), "
            "data_seed (synthetic-truth/noise seed), rng_seed, init_label "
            "(initial label, 0 or 1). Samples latent labels z via a "
            "Swendsen-Wang cluster sweep (ising_cluster_block) with a "
            "precomputed Gaussian emission field + partial decoupling.")
        .method("step", (void (IsingHiddenPotts::*)())    &IsingHiddenPotts::step, "Run one sweep.")
        .method("step", (void (IsingHiddenPotts::*)(int)) &IsingHiddenPotts::step, "Run n sweeps.")
        .method("get_current", &IsingHiddenPotts::get_current)
        .method("set_current", &IsingHiddenPotts::set_current,
                "Overwrite z (length L*L, entries in {0,1}) from a named list. "
                "Unknown keys are silently ignored.")
        .method("predict_at",
                (AI4BayesCode::history_map (IsingHiddenPotts::*)(const AI4BayesCode::state_map&) const)
                    &IsingHiddenPotts::predict_at,
                "predict_at(list()) returns list(y_rep = matrix): the posterior "
                "predictive y_rep_i ~ N(mu_{z_i}, sigma^2) at the N = L*L "
                "sites, with mu = (mu0, mu1) and sigma the fixed constructor "
                "values. keep_history = TRUE: one row per stored draw, drawn "
                "from the labels z of that draw (the same row of "
                "get_history()$z); one row, from the current labels, before "
                "the first sweep or right after clear_history(). "
                "keep_history = FALSE: one row, from the current labels. No "
                "covariate inputs: the list must be empty.")
        .method("predict_at",
                (AI4BayesCode::history_map (IsingHiddenPotts::*)(const AI4BayesCode::state_map&, bool) const)
                    &IsingHiddenPotts::predict_at,
                "predict_at(list(), last_draw_only): TRUE returns one row, "
                "drawn from the current labels, without discarding the "
                "history; FALSE is predict_at(list()).")
        .method("get_dag",     &IsingHiddenPotts::get_dag)
        .method("get_history", &IsingHiddenPotts::get_history)
        AI4BAYESCODE_BIND_KERNEL_CONTROL(IsingHiddenPotts);
}
#endif

#ifdef AI4BAYESCODE_PYBIND_MODULE
#include "AI4BayesCode/pybind_casters.hpp"
PYBIND11_MODULE(IsingHiddenPotts, m) {
    AI4BayesCode::register_ai4bayescode_types(m);
    pybind11::class_<IsingHiddenPotts>(m, "IsingHiddenPotts")
        .def(pybind11::init<int, double, double, double, double, double,
                            int, int, int, bool>(),
             pybind11::arg("L"),
             pybind11::arg("beta"),
             pybind11::arg("delta"),
             pybind11::arg("mu0"),
             pybind11::arg("mu1"),
             pybind11::arg("sigma"),
             pybind11::arg("data_seed"),
             pybind11::arg("rng_seed") = 1,
             pybind11::arg("init_label") = 0,
             pybind11::arg("keep_history") = false)
        .def("step", (void (IsingHiddenPotts::*)())    &IsingHiddenPotts::step, "Run one sweep.")
        .def("step", (void (IsingHiddenPotts::*)(int)) &IsingHiddenPotts::step, pybind11::arg("n_steps"))
        .def("get_current", &IsingHiddenPotts::get_current)
        .def("set_current", &IsingHiddenPotts::set_current, pybind11::arg("params"))
        .def("predict_at",
             (AI4BayesCode::history_map (IsingHiddenPotts::*)(const AI4BayesCode::state_map&, bool) const)
                 &IsingHiddenPotts::predict_at,
             pybind11::arg("new_data"),
             pybind11::arg("last_draw_only") = false,
             "predict_at({}, last_draw_only=False) returns {'y_rep': array}: "
             "the posterior predictive y_rep_i ~ N(mu_{z_i}, sigma^2) at the "
             "N = L*L sites, with mu = (mu0, mu1) and sigma the fixed "
             "constructor values. keep_history=True: one row per stored draw, "
             "drawn from the labels z of that draw (the same row of "
             "get_history()['z']). keep_history=False, or "
             "last_draw_only=True: one row, from the current labels. No "
             "covariate inputs: new_data must be empty.")
        .def("get_dag",     &IsingHiddenPotts::get_dag)
        .def("get_history", &IsingHiddenPotts::get_history)
        AI4BAYESCODE_PYBIND_KERNEL_CONTROL(IsingHiddenPotts);
}
#endif

#if !defined(AI4BAYESCODE_RCPP_MODULE) && !defined(AI4BAYESCODE_PYBIND_MODULE)
//==============================================================================
//  Standalone frontend-independent demo (active ONLY as a plain binary).
//  Checks segmentation accuracy (threshold posterior label marginals at 0.5)
//  and a two-chain Gelman-Rubin R-hat on the total up-count Σz.
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
            const auto gc = m.get_current();
            const arma::vec& z = gc.at("z");
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
#endif

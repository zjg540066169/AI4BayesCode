// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v3.0 or later
// (GPL-3.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
//  ICARSpatialGMRF_joint.cpp
//
//  JOINT-NUTS rewrite of ICARSpatialGMRF.cpp. Same model, same priors,
//  SAME posterior — but the three scalar hyperparameters (Intercept, tau,
//  sigma) are sampled by ONE joint_nuts_block instead of three separate
//  single nuts_blocks updated Gibbs-style. phi is still sampled by the
//  gmrf_precision_block (Rue 2001 direct draw + hard sum-to-zero),
//  unchanged.
//
//  Model (identical to ICARSpatialGMRF.cpp)
//  ----------------------------------------
//      y_t       ~ N(Intercept + phi_{node_idx[t]}, sigma)  t = 1..T
//      phi       ~ ICAR(tau) with sum(phi) = 0
//      Intercept ~ N(0, 100)                    // sd = 10, var = 100*100? No: N(0, 10^2) so var=100
//      tau       ~ Gamma(shape = 2, rate = 2)
//      sigma     ~ Half-Normal(0, sd(y))        // Gelman 2006 weakly-informative
//
//  JOINT natural-scale log-density over [Intercept, tau, sigma] (each term
//  exactly ONCE; the SSR term was shared between the Intercept and sigma
//  conditionals in the original — here it appears once as the full likelihood):
//
//    log p(Intercept, tau, sigma | y, phi) =
//        - T log σ  - SSR / (2 σ²)              # Gaussian likelihood
//      - Intercept² / (2 * 100)                  # N(0, 10²) prior on Intercept
//      + (a - 1) log τ - b τ                     # Gamma(2, 2) prior on tau
//      + (N-1)/2 log τ - τ/2 φᵀRφ               # ICAR normalizer + quadratic
//      - σ² / (2 σ_prior_var)                    # Half-Normal(0, sigma_prior_sd)
//
//  NOTE on the Intercept prior variance:
//    The original says "N(0, 100)" with sd=10, so var = 10^2 = 100.
//    intercept_natural_log_density uses constexpr double intercept_var = 100.0,
//    confirming var = 100, i.e. the full prior term is -Intercept^2 / 200 (= -Intercept^2 / (2*100)).
//
//  Gradients (natural scale):
//    d/d Intercept = sum_resid / σ² - Intercept / 100
//    d/d tau       = (a-1)/τ - b + (N-1)/(2τ) - φᵀRφ/2
//    d/d sigma     = -T/σ + SSR/σ³ - σ/σ_prior_var
//
//  Block decomposition
//  -------------------
//      phi             : gmrf_precision_block (Rue 2001 direct draw, sum-to-zero)
//      (Intercept, tau, sigma) : ONE joint_nuts_block
//                        sub_params = [{ "Intercept", 1, REAL },
//                                      { "tau",       1, POSITIVE },
//                                      { "sigma",     1, POSITIVE }]
//
//  cross-validation target: ICARSpatialGMRF.cpp (same model, same data,
//  posteriors must match under R-hat + means/sd/ESS comparison).
// ============================================================================

// @example:R
//   library(AI4BayesCode)
//   ai4bayescode_example("ICARSpatialGMRF")
//   P<-5; N<-P*P; cc<-(0:(N-1))%%P; rr<-(0:(N-1))%/%P                   # 5x5 lattice, N=25 nodes
//   ei<-c((1:N)[cc<P-1], (1:N)[rr<P-1]); ej<-c((1:N)[cc<P-1]+1, (1:N)[rr<P-1]+P)  # grid edges (1-based)
//   x<-cc/(P-1); yy<-rr/(P-1); phi<-2*sin(pi*x)*sin(pi*yy)+(x-0.5); phi<-phi-mean(phi)  # sum-to-zero field
//   node<-rep(1:N, each=5L)                                             # 5 replicate obs/node -> identifies sigma
//   set.seed(20260621); y<-4.0+phi[node]+rnorm(N*5L,0,0.5)             # Intercept=4, sigma=0.5
//   # ---- Recommended: parallel chains + convergence diagnosis ----
//   run <- ai4bayescode_run_chains(
//       function(seed) new(ICARSpatialGMRF, y, node, N, ei, ej, seed, TRUE),
//       n_chains = 4, n_burn = 1000, n_keep = 2000)
//   ai4bayescode_diagnose(run$histories[[1]])      # summary + R-hat/ESS + plots
//   # ---- Advanced: stateful single-chain control ----
//   m<-new(ICARSpatialGMRF, y, node, N, ei, ej, 7L, TRUE)             # y,node_idx,N,edge_i,edge_j,seed,keep_hist
//   m$step(2500); str(m$get_current())
// @example:python
//   import numpy as np, AI4BayesCode
//   P = 5; N = P*P
//   cc = np.arange(N) % P; rr = np.arange(N) // P                       # 5x5 lattice, N=25 nodes
//   ei = np.concatenate([np.where(cc < P-1)[0] + 1, np.where(rr < P-1)[0] + 1])           # grid edges (1-based)
//   ej = np.concatenate([np.where(cc < P-1)[0] + 2, np.where(rr < P-1)[0] + 1 + P])
//   x = cc/(P-1); yy = rr/(P-1); phi = 2*np.sin(np.pi*x)*np.sin(np.pi*yy) + (x-0.5)
//   phi = phi - phi.mean()                                              # sum-to-zero field
//   node = np.repeat(np.arange(1, N+1), 5)                              # 5 replicate obs/node -> identifies sigma
//   rng = np.random.default_rng(20260621)
//   y = 4.0 + phi[node-1] + rng.normal(0.0, 0.5, N*5)                   # Intercept=4, sigma=0.5
//   Mod = AI4BayesCode.example("ICARSpatialGMRF")
//   # ---- Recommended: parallel chains + diagnosis ----
//   chains = AI4BayesCode.run_chains(
//       lambda seed: Mod.ICARSpatialGMRF(y, node.astype(float), N, ei.astype(float), ej.astype(float), seed, True),
//       seeds=[101, 202, 303, 404], n_burn=1000, n_keep=2000, n_jobs=1)
//   AI4BayesCode.diagnose(chains[0]["hist"])   # summary + diagnostics
//   # ---- Advanced: stateful single-chain control ----
//   m = Mod.ICARSpatialGMRF(y, node.astype(float), N,                  # indices NUMERIC (no arma::ivec caster)
//                           ei.astype(float), ej.astype(float), 7, True)
//   m.step(2500); print(m.get_current())                               # Intercept~4, sigma~0.5
// @example:end

// [[Rcpp::depends(RcppArmadillo)]]

#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif

#ifdef AI4BAYESCODE_RCPP_MODULE
#include <RcppArmadillo.h>
#else
#include <armadillo>
#endif

#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/backend_neutral.hpp"
#include "AI4BayesCode/shared_data.hpp"
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/constraints.hpp"
#include "AI4BayesCode/joint_nuts_block.hpp"
#include "AI4BayesCode/gmrf_precision_block.hpp"
#include "AI4BayesCode/rcpp_wrap.hpp"
#include "AI4BayesCode/kernel_control_mixin.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>

using AI4BayesCode::block_context;
using AI4BayesCode::composite_block;
using AI4BayesCode::joint_nuts_block;
using AI4BayesCode::joint_nuts_block_config;
using AI4BayesCode::joint_nuts_sub_param;
using AI4BayesCode::joint_constraint;
using AI4BayesCode::gmrf_precision_block;
using AI4BayesCode::gmrf_precision_block_config;
namespace constraints = AI4BayesCode::constraints;

namespace {

// ----------------------------------------------------------------------------
//  Helper: build the (improper) graph Laplacian R = D - W.
//  (Identical to ICARSpatialGMRF.cpp; duplicated to keep files self-contained.)
// ----------------------------------------------------------------------------
arma::sp_mat
build_graph_laplacian(std::size_t N,
                       const arma::vec& edge_i,
                       const arma::vec& edge_j) {
    arma::sp_mat L(N, N);
    arma::vec    deg(N, arma::fill::zeros);
    const std::size_t E = edge_i.n_elem;
    for (std::size_t e = 0; e < E; ++e) {
        const std::size_t i = static_cast<std::size_t>(edge_i[e]);
        const std::size_t j = static_cast<std::size_t>(edge_j[e]);
        L(i, j) = -1.0;
        L(j, i) = -1.0;
        deg[i] += 1.0;
        deg[j] += 1.0;
    }
    for (std::size_t k = 0; k < N; ++k) L(k, k) = deg[k];
    return L;
}

// ----------------------------------------------------------------------------
//  Helper: phi^T R phi = Σ_{(i,j) ∈ E} (phi_i - phi_j)²  (Rue 2001 eq. 1)
// ----------------------------------------------------------------------------
double
phi_R_phi(const arma::vec& phi,
          const arma::vec& edge_i,
          const arma::vec& edge_j) {
    double s = 0.0;
    const std::size_t E = edge_i.n_elem;
    for (std::size_t e = 0; e < E; ++e) {
        const std::size_t i = static_cast<std::size_t>(edge_i[e]);
        const std::size_t j = static_cast<std::size_t>(edge_j[e]);
        const double d = phi[i] - phi[j];
        s += d * d;
    }
    return s;
}

// ----------------------------------------------------------------------------
//  Joint natural-scale log-density of (Intercept, tau, sigma) given y, phi.
//
//  theta_cat layout (NATURAL scale):
//      theta_cat[0] = Intercept  (real)
//      theta_cat[1] = tau        (positive)
//      theta_cat[2] = sigma      (positive)
//
//  joint_nuts_block handles Jacobians internally; this function must NOT
//  include them (exactly as the single-block versions do not include them).
//
//  Every model term appears EXACTLY ONCE:
//    - Gaussian likelihood:          -T log σ  - SSR/(2σ²)
//    - Intercept N(0, 100) prior:    -Intercept²/200
//    - tau Gamma(2,2) prior:         (a-1) log τ - b τ
//    - ICAR normalizer + quadratic:  (N-1)/2 log τ  - τ/2 φᵀRφ
//    - sigma Half-Normal prior:      -σ²/(2 σ_prior_var)
//
//  The SSR was shared between the two original conditionals; here it appears
//  only once inside the likelihood term.
// ----------------------------------------------------------------------------
double
icar_joint_log_density(const arma::vec& theta_cat,
                       const block_context& ctx,
                       arma::vec* grad_nat) {
    // Slice the concatenated natural-scale vector.
    const double Intercept = theta_cat[0];
    const double tau       = theta_cat[1];
    const double sigma     = theta_cat[2];

    // Guard: positivity of tau and sigma (joint_nuts_block enforces these
    // via its POSITIVE sub-param transforms, but be explicit for safety).
    if (tau <= 0.0 || sigma <= 0.0) {
        if (grad_nat) grad_nat->set_size(3);
        return -std::numeric_limits<double>::infinity();
    }

    // Read TRUE external data from ctx. phi, y, node_idx come from data(),
    // written by the gmrf_precision_block each step. Intercept, tau, sigma
    // are NOT read from ctx — they come from the concatenated vector above.
    const arma::vec& y              = ctx.at("y");
    const arma::vec& node_idx       = ctx.at("node_idx");
    const arma::vec& phi            = ctx.at("phi");
    const arma::vec& edge_i         = ctx.at("edge_i");
    const arma::vec& edge_j         = ctx.at("edge_j");
    const double     sigma_prior_sd = ctx.at("sigma_prior_sd")[0];
    const std::size_t N             = static_cast<std::size_t>(ctx.at("N")[0]);
    const std::size_t T             = y.n_elem;

    // Pre-compute quantities needed for lp and gradients.
    const double sigma2         = sigma * sigma;
    const double sigma3         = sigma2 * sigma;
    const double sigma_prior_var = sigma_prior_sd * sigma_prior_sd;
    const double Td             = static_cast<double>(T);
    const double Nd             = static_cast<double>(N);
    const double n_minus_1      = Nd - 1.0;

    // SSR and sum_resid (for grad wrt Intercept).
    double SSR       = 0.0;
    double sum_resid = 0.0;
    for (std::size_t t = 0; t < T; ++t) {
        const std::size_t k = static_cast<std::size_t>(node_idx[t]);
        const double r = y[t] - Intercept - phi[k];
        SSR       += r * r;
        sum_resid += r;
    }

    // phi^T R phi (needed by tau terms).
    const double pair_sum_sq = phi_R_phi(phi, edge_i, edge_j);

    // ---- Gamma(2,2) hyperparams -----------------------------------------
    constexpr double a_gamma = 2.0;
    constexpr double b_gamma = 2.0;

    // ---- Intercept prior variance ----------------------------------------
    constexpr double intercept_var = 100.0;  // N(0, 10^2): var = 100

    // ---- Assemble log p ---------------------------------------------------
    const double log_tau   = std::log(tau);
    const double log_sigma = std::log(sigma);

    const double lp =
        // Gaussian likelihood (full, including log σ term)
        - Td * log_sigma
        - 0.5 * SSR / sigma2
        // Intercept prior N(0, 100)
        - 0.5 * Intercept * Intercept / intercept_var
        // tau Gamma(2,2) prior
        + (a_gamma - 1.0) * log_tau - b_gamma * tau
        // ICAR normalizer + quadratic: p(phi|tau) normalizer contribution
        + 0.5 * n_minus_1 * log_tau
        - 0.5 * tau * pair_sum_sq
        // sigma Half-Normal prior
        - 0.5 * sigma2 / sigma_prior_var;

    if (grad_nat) {
        grad_nat->set_size(3);
        // d/d Intercept: likelihood contributes sum_resid/σ², prior contributes -Intercept/intercept_var
        (*grad_nat)[0] = sum_resid / sigma2 - Intercept / intercept_var;
        // d/d tau: Gamma prior + ICAR terms
        (*grad_nat)[1] = (a_gamma - 1.0) / tau - b_gamma
                          + 0.5 * n_minus_1 / tau
                          - 0.5 * pair_sum_sq;
        // d/d sigma: likelihood -T/σ + SSR/σ³, Half-Normal prior -σ/σ_prior_var
        (*grad_nat)[2] = -Td / sigma + SSR / sigma3 - sigma / sigma_prior_var;
    }

    return lp;
}

}  // anonymous namespace

class ICARSpatialGMRF : public AI4BayesCode::kernel_control_mixin<ICARSpatialGMRF> {
public:
    ICARSpatialGMRF(const arma::vec& y,
                         const arma::vec& node_idx_1based,   // 1-based node index
                                                             // per obs; passed as
                                                             // NUMERIC so both the
                                                             // R-list and Python-dict
                                                             // backend casters apply
                                                             // (no arma::ivec caster
                                                             // exists in pybind_casters)
                         int N_nodes,
                         const arma::vec& edges_i_1based,    // 1-based edge endpoints
                         const arma::vec& edges_j_1based,    // (numeric; see above)
                         int rng_seed,
                         bool keep_history = false)
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
          impl_(std::make_unique<composite_block>("ICARSpatialGMRF")),
          keep_history_(keep_history)
    {
        const std::size_t T = y.n_elem;
        const std::size_t N = static_cast<std::size_t>(N_nodes);
        const std::size_t E = edges_i_1based.n_elem;
        if (edges_j_1based.n_elem != E) {
            ai4b::stop("edges_i and edges_j must have the same length");
        }
        if (N < 2) ai4b::stop("N_nodes must be >= 2");
        if (E < 1) ai4b::stop("E must be >= 1");
        if (node_idx_1based.n_elem != T) {
            ai4b::stop("node_idx must have same length as y");
        }

        // Convert 1-based → 0-based. Indices arrive as NUMERIC (double); round
        // to the nearest integer before subtracting 1 (cf. HierarchicalLM_joint).
        arma::vec node_idx(T);
        for (std::size_t t = 0; t < T; ++t) {
            const int k = static_cast<int>(std::llround(node_idx_1based[t])) - 1;
            if (k < 0 || k >= static_cast<int>(N)) {
                ai4b::stop("node_idx entries must be in 1..N_nodes");
            }
            node_idx[t] = static_cast<double>(k);
        }
        arma::vec edge_i(E), edge_j(E);
        for (std::size_t e = 0; e < E; ++e) {
            const int i0 = static_cast<int>(std::llround(edges_i_1based[e])) - 1;
            const int j0 = static_cast<int>(std::llround(edges_j_1based[e])) - 1;
            if (i0 < 0 || j0 < 0 ||
                i0 >= static_cast<int>(N) || j0 >= static_cast<int>(N)) {
                ai4b::stop("edge endpoints must be in 1..N_nodes");
            }
            edge_i[e] = static_cast<double>(i0);
            edge_j[e] = static_cast<double>(j0);
        }

        // Build R and per-node observation counts once.
        R_           = build_graph_laplacian(N, edge_i, edge_j);
        node_counts_ = arma::vec(N, arma::fill::zeros);
        for (std::size_t t = 0; t < T; ++t) {
            const std::size_t k = static_cast<std::size_t>(node_idx[t]);
            node_counts_[k] += 1.0;
        }
        N_ = N;
        T_ = T;

        const double y_sd           = arma::stddev(y);
        const double sigma_prior_sd = (y_sd > 0.0) ? y_sd : 1.0;

        // Initial values (natural scale).
        const double Intercept_init = arma::mean(y);
        const double tau_init       = 1.0;
        const double sigma_init     = sigma_prior_sd;

        // ---- shared_data ---------------------------------------------------
        impl_->data().set("y",              y);
        impl_->data().set("node_idx",       node_idx);
        impl_->data().set("edge_i",         edge_i);
        impl_->data().set("edge_j",         edge_j);
        impl_->data().set("N",              arma::vec{static_cast<double>(N)});
        impl_->data().set("T",              arma::vec{static_cast<double>(T)});
        impl_->data().set("E",              arma::vec{static_cast<double>(E)});
        impl_->data().set("sigma_prior_sd", arma::vec{sigma_prior_sd});
        // Sub-param slots: joint_nuts_block writes back under these names.
        impl_->data().set("Intercept", arma::vec{Intercept_init});
        impl_->data().set("tau",       arma::vec{tau_init});
        impl_->data().set("sigma",     arma::vec{sigma_init});
        impl_->data().set("phi",       arma::vec(N, arma::fill::zeros));

        // ---- Gibbs DAG declarations ----------------------------------------
        // phi's dependencies (for gmrf_precision_block): it reads y, node_idx,
        // Intercept, tau, sigma from shared_data.
        impl_->data().declare_dependencies(
            "phi", {"y", "node_idx", "Intercept", "tau", "sigma"});
        // Joint block's dependencies are keyed under the JOINT BLOCK NAME.
        // They are the union of what Intercept, tau, sigma individually need
        // from data(), MINUS the cross-reads of each other (those are now
        // internal to the joint block's concatenated vector):
        //   - Intercept needs: y, node_idx, phi  (sigma read from cat)
        //   - tau needs: phi, edge_i, edge_j, N  (no y, sigma)
        //   - sigma needs: y, node_idx, phi       (Intercept read from cat)
        //   Union external: y, node_idx, phi, edge_i, edge_j, N, sigma_prior_sd
        impl_->data().declare_dependencies(
            "params_joint",
            {"y", "node_idx", "phi", "edge_i", "edge_j", "N", "sigma_prior_sd"});

        // ---- gmrf_precision_block for phi ----------------------------------
        // R and node_counts captured by copy to avoid lifetime issues.
        const arma::sp_mat R_copy      = R_;
        const arma::vec    counts_copy = node_counts_;
        const std::size_t  N_cap       = N_;

        gmrf_precision_block_config gmrf_cfg;
        gmrf_cfg.name = "phi";
        gmrf_cfg.n    = N_;
        gmrf_cfg.Q_fn =
            [R_copy, counts_copy, N_cap](const block_context& ctx2)
            -> arma::sp_mat {
                const double tau2    = ctx2.at("tau")[0];
                const double sigma2v = ctx2.at("sigma")[0];
                const double s2      = sigma2v * sigma2v;
                arma::sp_mat Q = tau2 * R_copy;
                for (std::size_t i = 0; i < N_cap; ++i) {
                    Q(i, i) += counts_copy[i] / s2;
                }
                return Q;
            };
        gmrf_cfg.b_fn =
            [N_cap](const block_context& ctx2) -> arma::vec {
                const arma::vec& y_v        = ctx2.at("y");
                const arma::vec& node_idx_v = ctx2.at("node_idx");
                const double     Intercept2 = ctx2.at("Intercept")[0];
                const double     sigma2v    = ctx2.at("sigma")[0];
                const double     s2         = sigma2v * sigma2v;
                arma::vec b(N_cap, arma::fill::zeros);
                const std::size_t T_loc = y_v.n_elem;
                for (std::size_t t = 0; t < T_loc; ++t) {
                    const std::size_t k =
                        static_cast<std::size_t>(node_idx_v[t]);
                    b[k] += (y_v[t] - Intercept2) / s2;
                }
                return b;
            };
        gmrf_cfg.sum_to_zero = true;
        impl_->add_child(
            std::make_unique<gmrf_precision_block>(std::move(gmrf_cfg)));

        // ---- ONE joint_nuts_block over (Intercept, tau, sigma) ------------
        {
            joint_nuts_block_config cfg;
            cfg.name = "params_joint";
            cfg.sub_params.push_back(
                joint_nuts_sub_param{ "Intercept", 1, joint_constraint::REAL });
            cfg.sub_params.push_back(
                joint_nuts_sub_param{ "tau",       1, joint_constraint::POSITIVE });
            cfg.sub_params.push_back(
                joint_nuts_sub_param{ "sigma",     1, joint_constraint::POSITIVE });
            // initial_cat is NATURAL-scale: [Intercept_init, tau_init, sigma_init].
            cfg.initial_cat = arma::vec{ Intercept_init, tau_init, sigma_init };
            cfg.log_density_grad = &icar_joint_log_density;
            // Intercept is real-valued (could be e.g. 0), tau and sigma are
            // positive with different scales. use_diagonal_metric is required
            // for faithfulness across sub-params with very different variances.
            cfg.use_diagonal_metric = true;
            // Give the joint block adequate warmup runway (3 params).
            cfg.n_warmup_first_call = 800;
            impl_->add_child(std::make_unique<joint_nuts_block>(std::move(cfg)));
        }

        if (keep_history_) impl_->set_keep_history(true);
    }

    // ---- Canonical 6-method R interface --------------------------------

    void step() { step(1); }              // no-arg convenience: one sweep
    void step(int n_steps) {
        if (n_steps < 0) ai4b::stop("n_steps must be >= 0");
        for (int i = 0; i < n_steps; ++i) impl_->step(rng_);
    }

    AI4BayesCode::state_map get_current() const {
        AI4BayesCode::state_map out;
        out["Intercept"] = impl_->data().get("Intercept");   // length-1 vec
        out["tau"]       = impl_->data().get("tau");          // length-1 vec
        out["sigma"]     = impl_->data().get("sigma");        // length-1 vec
        out["phi"]       = impl_->data().get("phi");          // length-N vec
        return out;
    }

    void set_current(const AI4BayesCode::state_map& params) {
        // The joint block is child(1); the gmrf block is child(0).
        auto& jblk = dynamic_cast<joint_nuts_block&>(impl_->child(1));
        arma::vec cat_new = jblk.current();  // [Intercept, tau, sigma]
        bool touched = false;

        if (params.count("Intercept")) {
            const arma::vec& v = params.at("Intercept");
            if (v.n_elem != 1) ai4b::stop("Intercept must be a length-1 vector");
            cat_new[0] = v[0];
            touched = true;
        }
        if (params.count("tau")) {
            const arma::vec& v = params.at("tau");
            if (v.n_elem != 1 || !(v[0] > 0.0))
                ai4b::stop("tau must be a length-1 strictly-positive vector");
            cat_new[1] = v[0];
            touched = true;
        }
        if (params.count("sigma")) {
            const arma::vec& v = params.at("sigma");
            if (v.n_elem != 1 || !(v[0] > 0.0))
                ai4b::stop("sigma must be a length-1 strictly-positive vector");
            cat_new[2] = v[0];
            touched = true;
        }
        if (touched) {
            jblk.set_current(cat_new);
            impl_->data().set("Intercept", arma::vec{cat_new[0]});
            impl_->data().set("tau",       arma::vec{cat_new[1]});
            impl_->data().set("sigma",     arma::vec{cat_new[2]});
        }
        if (params.count("phi")) {
            const arma::vec& v = params.at("phi");
            if (v.n_elem != N_)
                ai4b::stop("phi must have length N_nodes");
            impl_->data().set("phi", v);
            auto* blk = dynamic_cast<gmrf_precision_block*>(&impl_->child(0));
            if (blk) blk->set_current(v);
        }
        if (params.count("y")) {
            const arma::vec& yn = params.at("y");
            if (yn.n_elem != T_) ai4b::stop("y length must equal T");
            impl_->data().set("y", yn);
        }
    }

    AI4BayesCode::history_map predict_at(const AI4BayesCode::state_map& new_data) const {
        // No y_rep stochastic refresher registered; no covariate inputs.
        // Matches the original no-op convention: any non-empty input is an
        // error, and the return is an empty history_map (rcpp_wrap / pybind
        // glue render this as an empty R list / Python dict).
        if (!new_data.empty()) {
            ai4b::stop("ICARSpatialGMRF::predict_at: no registered "
                       "refreshers (no y_rep in v0 scope).");
        }
        return AI4BayesCode::history_map{};
    }

    AI4BayesCode::dag_info     get_dag()     const { return impl_->get_dag();     }
    AI4BayesCode::history_map  get_history() const { return impl_->get_history(); }

    void readapt_NUTS(int n, bool reset, int max_tree_depth = -1) {
        if (n < 0) ai4b::stop("n must be >= 0");
        impl_->readapt_NUTS(static_cast<std::size_t>(n), reset, readapt_rng_, max_tree_depth < 0 ? std::size_t(0) : static_cast<std::size_t>(max_tree_depth));
    }

private:
    std::mt19937_64                  rng_;
    mutable std::mt19937_64          predict_rng_;
    mutable std::mt19937_64          readapt_rng_;
    std::unique_ptr<composite_block> impl_;
    bool                             keep_history_ = false;

    arma::sp_mat                     R_;
    arma::vec                        node_counts_;
    std::size_t                      N_ = 0;
    std::size_t                      T_ = 0;
};

#ifdef AI4BAYESCODE_RCPP_MODULE
RCPP_MODULE(ICARSpatialGMRF_module) {
    Rcpp::class_<ICARSpatialGMRF>("ICARSpatialGMRF")
        .constructor<arma::vec, arma::vec, int, arma::vec, arma::vec, int>(
            "Legacy constructor; keep_history defaults to FALSE. Args: "
            "y, node_idx_1based, N_nodes, edges_i_1based, edges_j_1based, "
            "rng_seed.")
        .constructor<arma::vec, arma::vec, int, arma::vec, arma::vec, int, bool>(
            "JOINT-NUTS ICARSpatialGMRF: y (T-length observation vec), "
            "node_idx_1based (T-length 1-based node index per obs), N_nodes "
            "(graph node count), edges_i_1based / edges_j_1based (1-based "
            "edge endpoints; undirected), rng_seed, keep_history. "
            "gmrf_precision_block for phi (Rue 2001 direct draw + hard "
            "sum-to-zero) + ONE joint_nuts_block for (Intercept, tau, sigma).")
        .method("step", (void (ICARSpatialGMRF::*)())    &ICARSpatialGMRF::step, "Run one sweep.")
        .method("step", (void (ICARSpatialGMRF::*)(int)) &ICARSpatialGMRF::step, "Run n sweeps.")
        .method("get_current",  &ICARSpatialGMRF::get_current)
        .method("set_current",  &ICARSpatialGMRF::set_current,
                "Overwrite any subset of {Intercept, tau, sigma, phi, y} "
                "via a named list. Unknown keys silently ignored.")
        .method("predict_at",   &ICARSpatialGMRF::predict_at,
                "Returns empty list — no y_rep refresher registered.")
        .method("get_dag",      &ICARSpatialGMRF::get_dag)
        .method("get_history",  &ICARSpatialGMRF::get_history)
        .method("readapt_NUTS", &ICARSpatialGMRF::readapt_NUTS,
                "Re-tune NUTS dual-averaging state for the joint hyperparam "
                "block. Chain state preserved. Args: n (internal iters), "
                "reset (bool; reinit DA state).")
        AI4BAYESCODE_BIND_KERNEL_CONTROL(ICARSpatialGMRF);
}
#endif

#ifdef AI4BAYESCODE_PYBIND_MODULE
#include "AI4BayesCode/pybind_casters.hpp"
PYBIND11_MODULE(ICARSpatialGMRF, m) {
    AI4BayesCode::register_ai4bayescode_types(m);
    pybind11::class_<ICARSpatialGMRF>(m, "ICARSpatialGMRF")
        .def(pybind11::init<arma::vec, arma::vec, int,
                            arma::vec, arma::vec, int, bool>(),
             pybind11::arg("y"),
             pybind11::arg("node_idx_1based"),
             pybind11::arg("N_nodes"),
             pybind11::arg("edges_i_1based"),
             pybind11::arg("edges_j_1based"),
             pybind11::arg("rng_seed") = 1,
             pybind11::arg("keep_history") = false)
        .def("step", (void (ICARSpatialGMRF::*)())    &ICARSpatialGMRF::step, "Run one sweep.")
        .def("step", (void (ICARSpatialGMRF::*)(int)) &ICARSpatialGMRF::step,        pybind11::arg("n_steps"))
        .def("get_current",  &ICARSpatialGMRF::get_current)
        .def("set_current",  &ICARSpatialGMRF::set_current, pybind11::arg("params"))
        .def("predict_at",   &ICARSpatialGMRF::predict_at,  pybind11::arg("new_data"))
        .def("get_dag",      &ICARSpatialGMRF::get_dag)
        .def("get_history",  &ICARSpatialGMRF::get_history)
        .def("readapt_NUTS", &ICARSpatialGMRF::readapt_NUTS,
             pybind11::arg("n"), pybind11::arg("reset") = false, pybind11::arg("max_tree_depth") = -1)
        AI4BAYESCODE_PYBIND_KERNEL_CONTROL(ICARSpatialGMRF);
}
#endif

#if !defined(AI4BAYESCODE_RCPP_MODULE) && !defined(AI4BAYESCODE_PYBIND_MODULE)
//==============================================================================
//  Standalone FRONTEND-INDEPENDENT demo (pure C++; no Rcpp, no pybind).
//
//  Builds a 5x5 lattice graph (N=25 nodes, 4-neighbour adjacency), draws a
//  true spatial field phi_true from the ICAR prior (smoothed lattice surface,
//  centered to sum-to-zero), then simulates one observation per node:
//      y_t = Intercept_true + phi_true[node] + sigma_true * noise.
//  Fits the joint-NUTS + gmrf model and checks recovery of Intercept and sigma,
//  AND that the posterior-mean spatial field correlates with phi_true (the GMRF
//  block actually recovers the spatial structure, not just the scalars).
//==============================================================================
#include <cstdio>

namespace {

// Build a P x P lattice graph; return edges (1-based) and node count.
void build_lattice(int P,
                   std::vector<int>& ei,
                   std::vector<int>& ej,
                   int& N_nodes) {
    N_nodes = P * P;
    auto idx = [P](int r, int c) { return r * P + c + 1; };  // 1-based
    for (int r = 0; r < P; ++r) {
        for (int c = 0; c < P; ++c) {
            if (c + 1 < P) { ei.push_back(idx(r, c)); ej.push_back(idx(r, c + 1)); }
            if (r + 1 < P) { ei.push_back(idx(r, c)); ej.push_back(idx(r + 1, c)); }
        }
    }
}

double pearson_corr(const arma::vec& a, const arma::vec& b) {
    const double ma = arma::mean(a), mb = arma::mean(b);
    double cov = 0.0, va = 0.0, vb = 0.0;
    for (std::size_t i = 0; i < a.n_elem; ++i) {
        const double da = a[i] - ma, db = b[i] - mb;
        cov += da * db; va += da * da; vb += db * db;
    }
    return cov / std::sqrt(va * vb);
}

}  // anonymous namespace

int main() {
    // ---- Build a 5x5 lattice graph -------------------------------------
    const int P = 5;
    std::vector<int> ei_v, ej_v;
    int N_nodes = 0;
    build_lattice(P, ei_v, ej_v, N_nodes);
    const std::size_t N = static_cast<std::size_t>(N_nodes);
    const std::size_t E = ei_v.size();

    arma::vec edges_i(E), edges_j(E);   // numeric 1-based endpoints (see ctor)
    for (std::size_t e = 0; e < E; ++e) {
        edges_i[e] = static_cast<double>(ei_v[e]);
        edges_j[e] = static_cast<double>(ej_v[e]);
    }

    // ---- True parameters ------------------------------------------------
    const double Intercept_true = 4.0;
    const double sigma_true     = 0.5;

    // ---- True spatial field: smooth lattice surface, sum-to-zero -------
    arma::vec phi_true(N);
    for (int r = 0; r < P; ++r) {
        for (int c = 0; c < P; ++c) {
            // smooth deterministic surface (plane + bump) -> spatially coherent
            const double x = static_cast<double>(c) / (P - 1);
            const double yy = static_cast<double>(r) / (P - 1);
            phi_true[r * P + c] =
                2.0 * std::sin(3.14159265 * x) * std::sin(3.14159265 * yy)
                + 1.0 * (x - 0.5);
        }
    }
    phi_true -= arma::mean(phi_true);  // enforce sum-to-zero like the model

    // ---- Simulate one observation per node ------------------------------
    const std::size_t T = N;  // one obs per node
    arma::vec  node_idx_1based(T);   // numeric 1-based node index (see ctor)
    arma::vec  y(T);
    std::mt19937_64 sim_rng(20260621ULL);
    std::normal_distribution<double> noise(0.0, sigma_true);
    for (std::size_t t = 0; t < T; ++t) {
        const std::size_t k = t;  // node k
        node_idx_1based[t] = static_cast<double>(k) + 1.0;  // 1-based
        y[t] = Intercept_true + phi_true[k] + noise(sim_rng);
    }

    // ---- Fit -------------------------------------------------------------
    ICARSpatialGMRF model(y, node_idx_1based, N_nodes,
                          edges_i, edges_j,
                          /*rng_seed=*/7, /*keep_history=*/false);
    model.step(1000);  // warmup (joint block auto-warms on first call too)

    const int M = 2000;
    double Intercept_bar = 0.0, sigma_bar = 0.0;
    arma::vec phi_bar(N, arma::fill::zeros);
    for (int s = 0; s < M; ++s) {
        model.step(1);
        const auto cur = model.get_current();
        Intercept_bar += cur.at("Intercept")[0];
        sigma_bar     += cur.at("sigma")[0];
        phi_bar       += cur.at("phi");
    }
    Intercept_bar /= static_cast<double>(M);
    sigma_bar     /= static_cast<double>(M);
    phi_bar       /= static_cast<double>(M);

    // ---- Recovery checks -------------------------------------------------
    const double phi_corr = pearson_corr(phi_bar, phi_true);

    std::printf("ICARSpatialGMRF demo (5x5 lattice, N=%d nodes, %zu edges, T=%zu obs)\n",
                N_nodes, E, T);
    std::printf("  Intercept_hat = %.3f   (truth %.1f)\n", Intercept_bar, Intercept_true);
    std::printf("  sigma_hat     = %.3f   (truth %.1f)\n", sigma_bar, sigma_true);
    std::printf("  corr(phi_hat, phi_true) = %.3f   (spatial-field recovery)\n", phi_corr);

    const bool ok_intercept = std::abs(Intercept_bar - Intercept_true) < 0.4;
    const bool ok_sigma     = std::abs(sigma_bar - sigma_true)         < 0.4;
    const bool ok_phi       = phi_corr > 0.8;  // posterior field tracks truth
    const bool ok = ok_intercept && ok_sigma && ok_phi;

    if (ok) {
        std::printf("[demo PASS] joint-NUTS + GMRF recovers (Intercept, sigma) "
                    "and the spatial field\n");
    } else {
        std::printf("[demo FAIL] Intercept_ok=%d sigma_ok=%d phi_ok=%d\n",
                    static_cast<int>(ok_intercept),
                    static_cast<int>(ok_sigma),
                    static_cast<int>(ok_phi));
    }
    return ok ? 0 : 1;
}
#endif

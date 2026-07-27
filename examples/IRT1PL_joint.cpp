// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v3.0 or later
// (GPL-3.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
//  IRT1PL_joint_v2.cpp
//
//  One-parameter logistic IRT (Rasch) model with a NON-CENTERED
//  reparameterization (NCR) for the item-difficulty hierarchy.  This
//  resolves the funnel geometry in IRT1PL_joint.cpp, which samples
//  (theta, b) jointly but leaves sigma_b in a SEPARATE nuts_block —
//  splitting the (sigma_b, b) funnel across two Gibbs blocks.
//
//  Mode 1 fix (joint_nuts_failure.md): fold sigma_b INTO the joint block
//  with NCR so the sampler never has to cross the funnel boundary
//  between a separate sigma_b step and the b-step.
//
//  Model (unchanged from IRT1PL_joint.cpp)
//  ----------------------------------------
//      Y_ij | theta_i, b_j  ~ Bernoulli( sigma(theta_i - b_j) )
//      theta_i              ~ Normal(0, 1)
//      b_j | sigma_b        ~ Normal(0, sigma_b^2)
//      sigma_b              ~ Half-Normal(0, 1)
//
//  NCR reparameterization
//  ----------------------
//      b_j = sigma_b * z_b_j,   z_b_j ~ Normal(0, 1)
//
//  Block decomposition (ONE joint block — no separate sigma_b nuts_block)
//  -----------------------------------------------------------------------
//      joint_nuts_block "theta_zb_sigma_joint":
//          sub_params:
//              [0]  "theta"   dim N  REAL     (student ability)
//              [1]  "z_b"     dim J  REAL     (standardised item difficulty)
//              [2]  "sigma_b" dim 1  POSITIVE (item-difficulty scale)
//      concatenated natural vector layout:
//          theta_cat[0..N-1]     = theta_i
//          theta_cat[N..N+J-1]   = z_b_j
//          theta_cat[N+J]        = sigma_b          (> 0)
//
//  Derived natural parameter (reconstructed after each step):
//      b_j = sigma_b * z_b_j   (written to shared_data["b"])
//
//  Joint natural-scale log-density (each model term exactly ONCE):
//      lp = -0.5 sum_i theta_i^2                           (theta ~ N(0,1))
//         - 0.5 sum_j z_b_j^2                              (z_b ~ N(0,1))
//         - 0.5 sigma_b^2                                  (sigma_b ~ HalfNormal(0,1))
//         + sum_{(i,j) in Obs} [Y_ij*eta_ij - log1pexp(eta_ij)]  (likelihood)
//      where eta_ij = theta_i - sigma_b * z_b_j.
//
//  NO Jacobian for sigma_b — the POSITIVE slice in joint_nuts_block adds
//  +log(sigma_b) internally (system_design.md §10.1 / validator Check #5).
//
//  Gradients (natural scale):
//      d/d theta_i  =  sum_{j: (i,j) in Obs} [Y_ij - p_ij] - theta_i
//      d/d z_b_j    = -sigma_b * sum_{i: (i,j) in Obs} [Y_ij - p_ij] - z_b_j
//      d/d sigma_b  = -sum_{j} sum_{i: (i,j) in Obs} z_b_j*[Y_ij - p_ij] - sigma_b
//      where p_ij = sigmoid(eta_ij).
//
//  Cross-validation target: get_current()["b"] must match IRT1PL_joint's
//  get_current()["b"] in posterior moments (same generative model).
//
//  VALIDATOR CONTRACT
//  ------------------
//  Passes Check #11 (joint_nuts_block audit):
//    1. Grad slices: [0,N) = d/dtheta, [N,N+J) = d/dz_b, [N+J] = d/dsigma_b.
//    2. All priors present (theta, z_b, sigma_b) with correct signs.
//    3. All REAL sub-params + one POSITIVE sub-param: mixed scale allowed by
//       joint_constraint enum.
//    4. No Jacobian written — block handles POSITIVE slice.
//    5. Write-back offsets match sub-param layout.
//    6. dim assert: N+J+1 == initial_cat.n_elem.
//  Passes Check #25 (joint-NUTS NCR): sigma_b is in the SAME joint block as
//    z_b; no funnel split.
// ============================================================================

// @example:R
//   library(AI4BayesCode)
//   ai4bayescode_example("IRT1PL_joint")
//   set.seed(2026)
//   N <- 200L; J <- 12L; sigma_b <- 1.2          # students, items, item-difficulty scale
//   theta <- rnorm(N)                            # student abilities ~ N(0,1)
//   b     <- sigma_b * rnorm(J)                  # item difficulties b_j = sigma_b * z_b_j
//   eta   <- outer(theta, b, "-")                # eta_ij = theta_i - b_j   (N x J)
//   Y     <- matrix(as.numeric(runif(N * J) < 1 / (1 + exp(-eta))), N, J)  # Bernoulli responses
//   # ctor: Y (N x J), theta_init (len N), b_init (len J), sigma_b_init (>0), seed, keep_history
//   # ---- Recommended: parallel chains + convergence diagnosis ----
//   run <- ai4bayescode_run_chains(
//       function(seed) new(IRT1PL_joint, Y, numeric(N), numeric(J), 1.0, seed, TRUE),
//       n_chains = 4, n_burn = 1000, n_keep = 2000)
//   ai4bayescode_diagnose(run$histories[[1]])      # summary + R-hat/ESS + plots
//   # ---- Advanced: stateful single-chain control ----
//   m <- new(IRT1PL_joint, Y, numeric(N), numeric(J), 1.0, 7L, TRUE)
//   m$step(2500); str(m$get_current())
// @example:python
//   import numpy as np, AI4BayesCode
//   rng = np.random.default_rng(2026)
//   N, J, sigma_b = 200, 12, 1.2                 # students, items, item-difficulty scale
//   theta = rng.standard_normal(N)               # student abilities ~ N(0,1)
//   b     = sigma_b * rng.standard_normal(J)     # item difficulties b_j = sigma_b * z_b_j
//   eta   = theta[:, None] - b[None, :]          # eta_ij = theta_i - b_j   (N x J)
//   Y     = (rng.random((N, J)) < 1 / (1 + np.exp(-eta))).astype(float)    # Bernoulli responses
//   Mod = AI4BayesCode.example("IRT1PL_joint")
//   # ctor: Y, theta_init (len N), b_init (len J), sigma_b_init (>0), seed, keep_history
//   # ---- Recommended: parallel chains + diagnosis ----
//   chains = AI4BayesCode.run_chains(
//       lambda seed: Mod.IRT1PL_joint(Y, np.zeros(N), np.zeros(J), 1.0, seed, True),
//       seeds=[101, 202, 303, 404], n_burn=1000, n_keep=2000, n_jobs=1)
//   AI4BayesCode.diagnose(chains[0]["hist"])   # summary + diagnostics
//   # ---- Advanced: stateful single-chain control ----
//   m = Mod.IRT1PL_joint(Y, np.zeros(N), np.zeros(J), 1.0, 7, True)
//   m.step(2500); print(m.get_current())
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
#include "AI4BayesCode/joint_nuts_block.hpp"
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/constraints.hpp"
#include "AI4BayesCode/rcpp_wrap.hpp"
#include "AI4BayesCode/kernel_control_mixin.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>

using AI4BayesCode::block_context;
using AI4BayesCode::composite_block;
using AI4BayesCode::joint_nuts_block;
using AI4BayesCode::joint_nuts_block_config;
using AI4BayesCode::joint_nuts_sub_param;
using AI4BayesCode::joint_constraint;

namespace constraints = AI4BayesCode::constraints;

// ============================================================================
//  Numerically stable helpers.
// ============================================================================

namespace {

inline double log1pexp(double x) {
    if (x > 0.0) return x + std::log1p(std::exp(-x));
    return std::log1p(std::exp(x));
}

inline double sigmoid(double x) {
    if (x >= 0.0) {
        const double e = std::exp(-x);
        return 1.0 / (1.0 + e);
    }
    const double e = std::exp(x);
    return e / (1.0 + e);
}

// ----------------------------------------------------------------------------
//  Joint natural-scale log-density for (theta, z_b, sigma_b) under NCR.
//
//  theta_cat layout:
//      [0 .. N-1]   = theta_i     (student abilities, REAL)
//      [N .. N+J-1] = z_b_j       (standardised item difficulties, REAL)
//      [N+J]        = sigma_b     (item-difficulty scale, POSITIVE — block enforces > 0)
//
//  lp = -0.5*sum theta_i^2                        (theta ~ N(0,1))
//       -0.5*sum z_b_j^2                           (z_b ~ N(0,1))
//       -0.5*sigma_b^2                             (sigma_b ~ HalfNormal(0,1))
//       + sum_{observed (i,j)} [Y_ij * eta_ij - log1pexp(eta_ij)]
//  where  eta_ij = theta_i - sigma_b * z_b_j.
//
//  NO +log(sigma_b) Jacobian here — the POSITIVE slice in joint_nuts_block
//  adds it automatically (validator Check #5).
//
//  Gradient (natural scale):
//      d/d theta_i  =  sum_{j: obs} (Y_ij - p_ij) - theta_i
//      d/d z_b_j    = -sigma_b * sum_{i: obs} (Y_ij - p_ij) - z_b_j
//      d/d sigma_b  = -sum_{j} z_b_j * sum_{i: obs} (Y_ij - p_ij) - sigma_b
// ----------------------------------------------------------------------------
double joint_theta_zb_sigma_log_density(const arma::vec& theta_cat,
                                        const block_context& ctx,
                                        arma::vec* grad_nat) {
    const arma::vec& Y_flat = ctx.at("Y");
    const arma::vec& M_flat = ctx.at("M");

    const std::size_t N = static_cast<std::size_t>(ctx.at("N")[0]);
    const std::size_t J = static_cast<std::size_t>(ctx.at("J")[0]);

    if (theta_cat.n_elem != N + J + 1) {
        // Defensive — dim-assert in joint_nuts_block constructor is primary.
        if (grad_nat) grad_nat->set_size(N + J + 1);
        return -std::numeric_limits<double>::infinity();
    }

    // Slice views (offsets match sub_params declaration order).
    auto theta_slice = theta_cat.subvec(0,     N - 1);
    auto zb_slice    = theta_cat.subvec(N,     N + J - 1);
    const double sigma_b = theta_cat[N + J];

    // sigma_b > 0 is guaranteed by the POSITIVE constraint in the block,
    // but guard defensively in case the function is called during boundary
    // exploration.
    if (sigma_b <= 0.0) {
        if (grad_nat) grad_nat->set_size(N + J + 1);
        return -std::numeric_limits<double>::infinity();
    }

    double lp = 0.0;

    if (grad_nat) {
        grad_nat->set_size(N + J + 1);
        grad_nat->zeros();

        // Priors.
        //   theta_i ~ N(0, 1):  -0.5*theta_i^2, d/dtheta_i = -theta_i
        //   z_b_j   ~ N(0, 1):  -0.5*z_b_j^2,   d/dz_b_j   = -z_b_j
        //   sigma_b ~ HalfNormal(0, 1): -0.5*sigma_b^2, d/dsigma_b = -sigma_b
        for (std::size_t i = 0; i < N; ++i) {
            const double th = theta_slice[i];
            lp             += -0.5 * th * th;
            (*grad_nat)[i]  = -th;
        }
        for (std::size_t j = 0; j < J; ++j) {
            const double zj   = zb_slice[j];
            lp               += -0.5 * zj * zj;
            (*grad_nat)[N + j] = -zj;
        }
        lp                    += -0.5 * sigma_b * sigma_b;
        (*grad_nat)[N + J]     = -sigma_b;   // d/d sigma_b from prior

        // Likelihood (observed entries only).
        for (std::size_t j = 0; j < J; ++j) {
            const double zj    = zb_slice[j];
            const double bj    = sigma_b * zj;          // b_j = sigma_b * z_b_j
            const std::size_t o = j * N;                // column-major offset
            for (std::size_t i = 0; i < N; ++i) {
                if (M_flat[o + i] == 0.0) continue;
                const double y    = Y_flat[o + i];
                const double eta  = theta_slice[i] - bj;
                lp               += y * eta - log1pexp(eta);
                const double resid = y - sigmoid(eta);
                (*grad_nat)[i]      += resid;             // d/d theta_i
                (*grad_nat)[N + j]  -= sigma_b * resid;  // d/d z_b_j (chain: d bj/d zj = sigma_b)
                (*grad_nat)[N + J]  -= zj * resid;       // d/d sigma_b (chain: d bj/d sigma_b = zj)
            }
        }
    } else {
        // No-gradient path (evaluation only).
        for (std::size_t i = 0; i < N; ++i) {
            const double th = theta_slice[i];
            lp += -0.5 * th * th;
        }
        for (std::size_t j = 0; j < J; ++j) {
            const double zj = zb_slice[j];
            lp += -0.5 * zj * zj;
        }
        lp += -0.5 * sigma_b * sigma_b;

        for (std::size_t j = 0; j < J; ++j) {
            const double bj    = sigma_b * zb_slice[j];
            const std::size_t o = j * N;
            for (std::size_t i = 0; i < N; ++i) {
                if (M_flat[o + i] == 0.0) continue;
                const double y   = Y_flat[o + i];
                const double eta = theta_slice[i] - bj;
                lp += y * eta - log1pexp(eta);
            }
        }
    }

    if (!std::isfinite(lp)) {
        return -std::numeric_limits<double>::infinity();
    }
    return lp;
}

} // anonymous namespace

// ============================================================================
//  User-facing class exposed to R.
// ============================================================================

class IRT1PL_joint : public AI4BayesCode::kernel_control_mixin<IRT1PL_joint> {
    friend class AI4BayesCode::kernel_control_mixin<IRT1PL_joint>;
public:
    IRT1PL_joint(const arma::mat& Y_input,
                   const arma::vec& theta_init,
                   const arma::vec& b_init,
                   double sigma_b_init,
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
          impl_(std::make_unique<composite_block>("IRT1PL_joint")),
          keep_history_(keep_history)
    {
        const std::size_t N = Y_input.n_rows;
        const std::size_t J = Y_input.n_cols;

        if (N < 1) throw std::runtime_error("Y must have at least 1 row");
        if (J < 1) throw std::runtime_error("Y must have at least 1 column");
        if (theta_init.n_elem != N)
            throw std::runtime_error("theta_init length must equal nrow(Y)");
        if (b_init.n_elem != J)
            throw std::runtime_error("b_init length must equal ncol(Y)");
        if (sigma_b_init <= 0.0)
            throw std::runtime_error("sigma_b_init must be > 0");

        // ---- Build Y and observation mask. NaN entries -> missing. -------
        arma::mat Y_clean(N, J, arma::fill::zeros);
        arma::mat M_mask (N, J, arma::fill::zeros);
        std::size_t n_obs = 0;
        for (std::size_t j = 0; j < J; ++j) {
            for (std::size_t i = 0; i < N; ++i) {
                const double y = Y_input(i, j);
                if (std::isnan(y)) continue;
                if (y != 0.0 && y != 1.0)
                    throw std::runtime_error("Y must contain only 0, 1, or NA");
                Y_clean(i, j) = y;
                M_mask (i, j) = 1.0;
                ++n_obs;
            }
        }
        if (n_obs == 0) throw std::runtime_error("Y has no observed entries");

        // ---- NCR initial values ------------------------------------------
        // z_b_j = b_j / sigma_b_init (standardised initial difficulties).
        arma::vec z_b_init(J);
        for (std::size_t j = 0; j < J; ++j)
            z_b_init[j] = b_init[j] / sigma_b_init;

        // ---- shared_data --------------------------------------------------
        impl_->data().set("Y",       arma::vectorise(Y_clean));
        impl_->data().set("M",       arma::vectorise(M_mask));
        impl_->data().set("N",       arma::vec{static_cast<double>(N)});
        impl_->data().set("J",       arma::vec{static_cast<double>(J)});

        // Natural parameters for output.
        impl_->data().set("theta",   theta_init);
        impl_->data().set("z_b",     z_b_init);
        impl_->data().set("sigma_b", arma::vec{sigma_b_init});
        // Derived: b = sigma_b * z_b  (kept current by a deterministic
        // refresher declared below; needed by y_rep and cross-validation).
        impl_->data().set("b",       b_init);

        // ---- Deterministic refresher: b = sigma_b * z_b ------------------
        // Runs after each joint_nuts_block update (triggered by
        // declare_invalidates on the joint block name), so downstream blocks
        // and generated quantities always read the current b.
        impl_->data().register_refresher(
            "b",
            [J](const AI4BayesCode::shared_data_t& d) -> arma::vec {
                const double sigma_b   = d.get("sigma_b")[0];
                const arma::vec& z_b   = d.get("z_b");
                arma::vec b(J);
                for (std::size_t j = 0; j < J; ++j)
                    b[j] = sigma_b * z_b[j];
                return b;
            });

        // ---- Dependency DAG ----------------------------------------------
        // The JOINT block reads Y, M, N, J from data; theta/z_b/sigma_b
        // are INTERNAL to the block. Key under the JOINT BLOCK NAME.
        impl_->data().declare_dependencies(
            "theta_zb_sigma_joint", {"Y", "M", "N", "J"});
        // b is derived from z_b + sigma_b; the joint block invalidates it.
        impl_->data().declare_invalidates(
            "theta_zb_sigma_joint", {"b"});

        // ---- Predict DAG + y_rep stochastic refresher --------------------
        // Sub-param names for predict edges (not the block name).
        // The y_rep refresher reads "b" (derived, always current), which
        // transitively depends on z_b and sigma_b.
        impl_->data().declare_predict_edges("theta", {"y_rep"});
        impl_->data().declare_predict_edges("b",     {"y_rep"});

        // ---- Generative-DAG context (VIZ-ONLY). --------------------------
        // sigma_b is the prior parent of z_b (conceptually); b = sigma_b*z_b
        // is the natural-scale item difficulty. theta has N(0,1) hardcoded.
        impl_->data().declare_context_edges("sigma_b", {"b"});
        impl_->data().declare_context_edges("z_b",     {"b"});

        impl_->data().set("y_rep", arma::vec(N * J, arma::fill::zeros));
        impl_->data().register_stochastic_refresher(
            "y_rep",
            [N, J](const AI4BayesCode::shared_data_t& d,
                   std::mt19937_64& rng) {
                const arma::vec& theta = d.get("theta");
                const arma::vec& b     = d.get("b");  // natural-scale b
                std::uniform_real_distribution<double> unif(0.0, 1.0);
                arma::vec y_rep(N * J);
                // Column-major: y_rep[i + j*N] corresponds to (i, j).
                for (std::size_t j = 0; j < J; ++j) {
                    for (std::size_t i = 0; i < N; ++i) {
                        const double eta = theta[i] - b[j];
                        const double p   = 1.0 / (1.0 + std::exp(-eta));
                        y_rep[i + j * N] = (unif(rng) < p) ? 1.0 : 0.0;
                    }
                }
                return y_rep;
            });

        // ---- ONE joint_nuts_block over (theta, z_b, sigma_b) -------------
        {
            joint_nuts_block_config cfg;
            cfg.name = "theta_zb_sigma_joint";

            cfg.sub_params.push_back(
                joint_nuts_sub_param{ "theta",   N, joint_constraint::REAL     });
            cfg.sub_params.push_back(
                joint_nuts_sub_param{ "z_b",     J, joint_constraint::REAL     });
            cfg.sub_params.push_back(
                joint_nuts_sub_param{ "sigma_b", 1, joint_constraint::POSITIVE });

            // Concatenated initial value (NATURAL scale):
            // [theta_init (N), z_b_init (J), sigma_b_init (1)].
            arma::vec init_cat(N + J + 1);
            for (std::size_t i = 0; i < N; ++i) init_cat[i]         = theta_init[i];
            for (std::size_t j = 0; j < J; ++j) init_cat[N + j]     = z_b_init[j];
            init_cat[N + J] = sigma_b_init;
            cfg.initial_cat = init_cat;

            cfg.log_density_grad = &joint_theta_zb_sigma_log_density;

            // NCR removes the funnel geometry — diagonal metric is sufficient.
            // More warmup runway than per-param NUTS since dim = N+J+1.
            cfg.use_diagonal_metric  = true;
            cfg.n_warmup_first_call  = 800;

            impl_->add_child(
                std::make_unique<joint_nuts_block>(std::move(cfg)));
        }

        if (keep_history_) impl_->set_keep_history(true);
    }

    void step() { step(1); }              // no-arg convenience: one sweep
    void step(int n_steps) {
        for (int i = 0; i < n_steps; ++i) impl_->step(rng_);
    }

    // get_current() exposes natural-scale (theta, b, sigma_b) to match
    // IRT1PL_joint's interface for cross-validation.
    AI4BayesCode::state_map get_current() const {
        AI4BayesCode::state_map out;
        out["theta"]   = impl_->data().get("theta");
        out["b"]       = impl_->data().get("b");       // derived: sigma_b * z_b
        out["sigma_b"] = impl_->data().get("sigma_b");
        return out;
    }

    // Also expose z_b for diagnostics (not part of the cross-val interface).
    AI4BayesCode::state_map get_current_raw() const {
        AI4BayesCode::state_map out;
        out["theta"]   = impl_->data().get("theta");
        out["z_b"]     = impl_->data().get("z_b");
        out["sigma_b"] = impl_->data().get("sigma_b");
        out["b"]       = impl_->data().get("b");
        return out;
    }

    void set_current(const AI4BayesCode::state_map& params) {
        auto& jblk = dynamic_cast<joint_nuts_block&>(impl_->child(0));
        const std::size_t N = jblk.sub_param_dim("theta");
        const std::size_t J = jblk.sub_param_dim("z_b");

        arma::vec cat_new = jblk.current();   // [theta (N), z_b (J), sigma_b]
        bool touched_cat  = false;

        // Accept theta directly.
        auto it_th = params.find("theta");
        if (it_th != params.end()) {
            const arma::vec& theta_new = it_th->second;
            if (theta_new.n_elem != N)
                throw std::runtime_error("theta has wrong length");
            for (std::size_t i = 0; i < N; ++i) cat_new[i] = theta_new[i];
            touched_cat = true;
        }

        // Accept natural b (convert to z_b = b / sigma_b).
        // sigma_b may itself be updated in the same call; handle order:
        // read sigma_b from params first if present, else from cat_new.
        double sigma_b_cur = cat_new[N + J];
        auto it_sb = params.find("sigma_b");
        if (it_sb != params.end()) {
            const double sb_new = it_sb->second[0];
            if (sb_new <= 0.0)
                throw std::runtime_error("sigma_b must be > 0");
            cat_new[N + J] = sb_new;
            sigma_b_cur    = sb_new;
            touched_cat    = true;
        }

        auto it_b = params.find("b");
        if (it_b != params.end()) {
            const arma::vec& b_new = it_b->second;
            if (b_new.n_elem != J)
                throw std::runtime_error("b has wrong length");
            if (sigma_b_cur <= 0.0)
                throw std::runtime_error("set_current: sigma_b <= 0 while setting b");
            for (std::size_t j = 0; j < J; ++j)
                cat_new[N + j] = b_new[j] / sigma_b_cur;  // z_b = b / sigma_b
            touched_cat = true;
        }

        // Accept z_b directly (lower-level; takes precedence over b).
        auto it_zb = params.find("z_b");
        if (it_zb != params.end()) {
            const arma::vec& zb_new = it_zb->second;
            if (zb_new.n_elem != J)
                throw std::runtime_error("z_b has wrong length");
            for (std::size_t j = 0; j < J; ++j)
                cat_new[N + j] = zb_new[j];
            touched_cat = true;
        }

        if (touched_cat) {
            jblk.set_current(cat_new);
            // Sync shared_data with the new natural values.
            impl_->data().set("theta",   cat_new.subvec(0,     N - 1));
            impl_->data().set("z_b",     cat_new.subvec(N,     N + J - 1));
            impl_->data().set("sigma_b", arma::vec{cat_new[N + J]});
            // Recompute derived b.
            const double sb = cat_new[N + J];
            arma::vec b_new(J);
            for (std::size_t j = 0; j < J; ++j)
                b_new[j] = sb * cat_new[N + j];
            impl_->data().set("b", b_new);
        }
    }

    // No covariate inputs; predict_at takes an empty map and returns
    // posterior-predictive y_rep as a 1 x (N*J) matrix (column-major
    // vectorisation of the N x J response matrix).
    AI4BayesCode::history_map predict_at(
            const AI4BayesCode::state_map& new_data) const {
        if (!new_data.empty()) {
            throw std::runtime_error(
                "IRT1PL_joint has no covariate inputs.");
        }
        AI4BayesCode::history_map out;

        if (!keep_history_) {
            block_context replaced;
            block_context result = impl_->predict_at(replaced, predict_rng_);
            for (const auto& kv : result) {
                arma::mat m(1, kv.second.n_elem);
                for (std::size_t j = 0; j < kv.second.n_elem; ++j)
                    m(0, j) = kv.second[j];
                out.emplace(kv.first, std::move(m));
            }
            return out;
        }

        // History mode: theta, z_b, sigma_b, b are sub-outputs of the
        // "theta_zb_sigma_joint" joint_nuts_block. Reconstruct b per-draw
        // and generate y_rep manually (same as IRT1PL_joint.cpp §history).
        AI4BayesCode::history_map hist = impl_->get_history();
        const arma::mat& theta_hist   = hist.at("theta");    // n_draws x N
        const arma::mat& zb_hist      = hist.at("z_b");      // n_draws x J
        const arma::mat& sigma_b_hist = hist.at("sigma_b");  // n_draws x 1
        const std::size_t n_draws = theta_hist.n_rows;
        const std::size_t N       = theta_hist.n_cols;
        const std::size_t J       = zb_hist.n_cols;

        arma::mat yrep(n_draws, N * J);
        std::uniform_real_distribution<double> unif(0.0, 1.0);
        for (std::size_t d = 0; d < n_draws; ++d) {
            const double sb = sigma_b_hist(d, 0);
            for (std::size_t j = 0; j < J; ++j) {
                const double bj = sb * zb_hist(d, j);         // b_j = sigma_b * z_b_j
                for (std::size_t i = 0; i < N; ++i) {
                    const double eta = theta_hist(d, i) - bj;
                    const double p   = 1.0 / (1.0 + std::exp(-eta));
                    yrep(d, i + j * N) = (unif(predict_rng_) < p) ? 1.0 : 0.0;
                }
            }
        }
        out.emplace("y_rep", std::move(yrep));
        return out;
    }

    AI4BayesCode::dag_info get_dag() const { return impl_->get_dag(); }

    AI4BayesCode::history_map get_history() const { return impl_->get_history(); }

    void readapt_NUTS(int n, bool reset = false, int max_tree_depth = -1) {
        if (n < 0) {
            ai4b::stop("readapt_NUTS: n must be non-negative");
        }
        impl_->readapt_NUTS(static_cast<std::size_t>(n), reset, readapt_rng_, max_tree_depth < 0 ? std::size_t(0) : static_cast<std::size_t>(max_tree_depth));
    }

private:
    std::mt19937_64                  rng_;
    mutable std::mt19937_64          predict_rng_;
    mutable std::mt19937_64          readapt_rng_;
    std::unique_ptr<composite_block> impl_;
    bool                             keep_history_ = false;
};

// ============================================================================
//  Rcpp module
// ============================================================================

#ifdef AI4BAYESCODE_RCPP_MODULE
RCPP_MODULE(IRT1PL_joint_module) {
    Rcpp::class_<IRT1PL_joint>("IRT1PL_joint")
        .constructor<arma::mat, arma::vec, arma::vec, double, int>(
            "Legacy constructor; keep_history defaults to FALSE.")
        .constructor<arma::mat, arma::vec, arma::vec, double, int, bool>(
            "NCR version: Y (N x J, NA for missing), theta_init (length N), "
            "b_init (length J), sigma_b_init (>0), RNG seed, keep_history. "
            "Joint block: (theta, z_b, sigma_b) with b = sigma_b*z_b derived.")
        .method("step", (void (IRT1PL_joint::*)())    &IRT1PL_joint::step, "Run one sweep.")
        .method("step", (void (IRT1PL_joint::*)(int)) &IRT1PL_joint::step, "Run n sweeps.")
        .method("get_current",     &IRT1PL_joint::get_current)
        .method("get_current_raw", &IRT1PL_joint::get_current_raw)
        .method("set_current",     &IRT1PL_joint::set_current)
        .method("predict_at",      &IRT1PL_joint::predict_at)
        .method("get_dag",         &IRT1PL_joint::get_dag)
        .method("get_history",     &IRT1PL_joint::get_history)
        .method("readapt_NUTS",    &IRT1PL_joint::readapt_NUTS)
        AI4BAYESCODE_BIND_KERNEL_CONTROL(IRT1PL_joint);
}
#endif

#ifdef AI4BAYESCODE_PYBIND_MODULE
#include "AI4BayesCode/pybind_casters.hpp"
PYBIND11_MODULE(IRT1PL_joint, m) {
    AI4BayesCode::register_ai4bayescode_types(m);
    pybind11::class_<IRT1PL_joint>(m, "IRT1PL_joint")
        .def(pybind11::init<arma::mat, arma::vec, arma::vec, double, int, bool>(),
             pybind11::arg("Y"),
             pybind11::arg("theta_init"),
             pybind11::arg("b_init"),
             pybind11::arg("sigma_b_init") = 1.0,
             pybind11::arg("rng_seed")     = 1,
             pybind11::arg("keep_history") = false,
             "NCR Rasch (1PL IRT) model: joint NUTS on (theta, z_b, sigma_b) "
             "with b = sigma_b * z_b reconstructed.")
        .def("step", (void (IRT1PL_joint::*)())    &IRT1PL_joint::step, "Run one sweep.")
        .def("step", (void (IRT1PL_joint::*)(int)) &IRT1PL_joint::step,
             pybind11::arg("n_steps"))
        .def("get_current",     &IRT1PL_joint::get_current)
        .def("get_current_raw", &IRT1PL_joint::get_current_raw)
        .def("set_current",     &IRT1PL_joint::set_current,
             pybind11::arg("params"))
        .def("predict_at",      &IRT1PL_joint::predict_at,
             pybind11::arg("new_data"))
        .def("get_dag",         &IRT1PL_joint::get_dag)
        .def("get_history",     &IRT1PL_joint::get_history)
        .def("readapt_NUTS",    &IRT1PL_joint::readapt_NUTS,
             pybind11::arg("n"), pybind11::arg("reset") = false, pybind11::arg("max_tree_depth") = -1)
        AI4BAYESCODE_PYBIND_KERNEL_CONTROL(IRT1PL_joint);
}
#endif

// ============================================================================
//  Standalone demo (only compiled when NEITHER frontend module is defined).
// ============================================================================
#if !defined(AI4BAYESCODE_RCPP_MODULE) && !defined(AI4BAYESCODE_PYBIND_MODULE)
//==============================================================================
//  Standalone FRONTEND-INDEPENDENT demo (pure C++; no Rcpp, no pybind).
//
//  Simulates a 1PL (Rasch) IRT dataset from a KNOWN truth:
//      theta_i ~ N(0,1)            (student abilities, i = 1..N)
//      b_j     = sigma_b * z_b_j   (item difficulties, j = 1..J)
//      Y_ij    ~ Bernoulli(sigmoid(theta_i - b_j))
//  then fits the joint-NUTS NCR sampler and checks posterior-mean recovery of
//  the item difficulties b (identifiable up to the global mean — abilities and
//  difficulties trade off an overall shift). We therefore compare the CENTERED
//  difficulties (b - mean(b)) against the centered truth, and require the
//  joint-NUTS estimate to beat a naive all-zero baseline.
//==============================================================================
#include <cstdio>

namespace {

// Correlation between two equal-length vectors (used for ability recovery,
// which is only identifiable up to an additive shift).
double corr_vec(const arma::vec& a, const arma::vec& b) {
    const double ma = arma::mean(a), mb = arma::mean(b);
    const arma::vec da = a - ma, db = b - mb;
    const double denom = std::sqrt(arma::dot(da, da) * arma::dot(db, db));
    if (denom <= 0.0) return 0.0;
    return arma::dot(da, db) / denom;
}

} // anonymous namespace

int main() {
    // ---- Simulation truth ----------------------------------------------------
    const std::size_t N         = 200;   // students
    const std::size_t J         = 12;    // items
    const double      sigma_b_t = 1.2;   // item-difficulty scale (truth)

    std::mt19937_64 sim_rng(12345);
    std::normal_distribution<double> rnorm(0.0, 1.0);
    std::uniform_real_distribution<double> runif(0.0, 1.0);

    arma::vec theta_true(N);
    for (std::size_t i = 0; i < N; ++i) theta_true[i] = rnorm(sim_rng);

    // b_j = sigma_b * z_b_j, z_b_j ~ N(0,1).
    arma::vec b_true(J);
    for (std::size_t j = 0; j < J; ++j) b_true[j] = sigma_b_t * rnorm(sim_rng);

    // Simulate the response matrix.
    arma::mat Y(N, J);
    for (std::size_t j = 0; j < J; ++j) {
        for (std::size_t i = 0; i < N; ++i) {
            const double eta = theta_true[i] - b_true[j];
            const double p   = 1.0 / (1.0 + std::exp(-eta));
            Y(i, j) = (runif(sim_rng) < p) ? 1.0 : 0.0;
        }
    }

    // ---- Inits (deliberately NOT the truth) ---------------------------------
    arma::vec theta_init(N, arma::fill::zeros);
    arma::vec b_init(J, arma::fill::zeros);
    const double sigma_b_init = 1.0;

    // ---- Fit -----------------------------------------------------------------
    IRT1PL_joint model(Y, theta_init, b_init, sigma_b_init,
                       /*rng_seed=*/7, /*keep_history=*/false);
    model.step(800);   // additional warmup beyond the block's first-call warmup

    const int M = 1500;
    arma::vec b_sum(J, arma::fill::zeros);
    arma::vec theta_sum(N, arma::fill::zeros);
    double    sigma_b_sum = 0.0;
    for (int s = 0; s < M; ++s) {
        model.step(1);
        const auto cur = model.get_current();
        b_sum       += cur.at("b");
        theta_sum   += cur.at("theta");
        sigma_b_sum += cur.at("sigma_b")[0];
    }
    const arma::vec b_hat     = b_sum / static_cast<double>(M);
    const arma::vec theta_hat = theta_sum / static_cast<double>(M);
    const double    sigma_b_hat = sigma_b_sum / static_cast<double>(M);

    // ---- Scoring -------------------------------------------------------------
    // Difficulties are identifiable only up to a global additive shift
    // (theta_i - b_j is invariant under theta+=c, b+=c). Compare CENTERED.
    const arma::vec b_true_c = b_true - arma::mean(b_true);
    const arma::vec b_hat_c  = b_hat  - arma::mean(b_hat);

    double rmse_fit = 0.0, rmse_naive = 0.0;
    for (std::size_t j = 0; j < J; ++j) {
        rmse_fit   += (b_hat_c[j]  - b_true_c[j]) * (b_hat_c[j]  - b_true_c[j]);
        rmse_naive +=  b_true_c[j] * b_true_c[j];   // naive: all difficulties = 0
    }
    rmse_fit   = std::sqrt(rmse_fit   / static_cast<double>(J));
    rmse_naive = std::sqrt(rmse_naive / static_cast<double>(J));

    const double theta_corr = corr_vec(theta_hat, theta_true);

    std::printf("IRT1PL_joint demo (N=%zu students, J=%zu items)\n", N, J);
    std::printf("  item difficulty b:  RMSE(fit)=%.3f  RMSE(naive 0)=%.3f"
                "  (centered)\n", rmse_fit, rmse_naive);
    std::printf("  ability theta:      corr(theta_hat, theta_true)=%.3f\n",
                theta_corr);
    std::printf("  sigma_b:            est=%.3f  (truth %.2f)\n",
                sigma_b_hat, sigma_b_t);

    // PASS: joint-NUTS materially beats the naive baseline on difficulty
    // recovery AND abilities are well-correlated with truth.
    const bool ok = (rmse_fit < 0.5 * rmse_naive) &&
                    (theta_corr > 0.7);
    std::printf("%s\n",
        ok ? "[demo PASS] joint-NUTS NCR recovers item difficulties & abilities"
           : "[demo FAIL]");
    return ok ? 0 : 1;
}
#endif

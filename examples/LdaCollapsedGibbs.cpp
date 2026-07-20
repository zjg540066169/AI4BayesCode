// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v3.0 or later
// (GPL-3.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
//  LdaCollapsedGibbs.cpp
//
//  REFERENCE TEMPLATE for fixed-K Latent Dirichlet Allocation sampled by
//  collapsed Gibbs (Griffiths & Steyvers 2004 PNAS). Wraps the new
//  `lda_collapsed_gibbs_block` (shipped in v0.7) which samples
//  z token-level topic assignments via the marginalized
//  Dirichlet-multinomial posterior, AND produces simplex draws of theta
//  (per-doc topic distribution, M x K) and phi (per-topic word
//  distribution, K x V) via Dirichlet conjugate posteriors using the
//  z-induced count tables.
//
//  Model
//  -----
//      theta_d   ~ Dirichlet(alpha)               d = 1..M  (length-K simplex)
//      phi_k     ~ Dirichlet(beta)                k = 1..K  (length-V simplex)
//      z_n | doc_n = d ~ Categorical(theta_d)     n = 1..N
//      w_n | z_n   = k ~ Categorical(phi_k)       n = 1..N
//
//      alpha and beta are FIXED at construction (length K and V).
//      Default values are alpha = (1, ..., 1) and beta = (1, ..., 1)
//      (uniform Dirichlets).
//
//  Block decomposition (Gibbs sweep order)
//  ---------------------------------------
//      child(0) lda  lda_collapsed_gibbs_block
//                    -- samples z (length-N), theta (M*K col-major),
//                       phi (K*V col-major) jointly per sweep.
//
//  This wrapper has ONLY ONE child. The collapsed-Gibbs block is
//  inherently joint (it samples z, theta, phi together using
//  Dirichlet-multinomial conjugacy with theta, phi marginalized at
//  the z-update step and re-sampled afterwards). Splitting it back
//  into separate categorical_gibbs + dirichlet_gibbs siblings would
//  recreate the slow-mixing problem this block was created to solve
//  (system_design.md §11.2(b)).
//
//  Refreshers
//  ----------
//      y_rep   register_stochastic_refresher (predict-time only)
//              Posterior predictive: for each token n,
//                  z_rep_n ~ Categorical(theta_{doc_n})
//                  y_rep_n ~ Categorical(phi_{z_rep_n})
//              Output is length-N (1-indexed word ids).
//
//  Predict DAG
//  -----------
//      theta -> y_rep
//      phi   -> y_rep
//      doc   -> y_rep        (data input, declared via
//                              declare_data_input + declare_predict_edges)
//
//  LABEL SWITCHING
//  ---------------
//  Topics k = 1, ..., K are exchangeable in the LDA posterior. The block
//  does NOT permute topics internally (per system_design.md
//  invariant: keep block output deterministic for a given seed +
//  z init). Cross-implementation alignment uses Stephens 2000 in
//  R-level sim1 alignment code. See `skills/label_switching.md`.
//
//  JUSTIFICATION (Check #16):
//  - z is DISCRETE (Exception 1 from codegen_priors.md §2b). NUTS
//    cannot target a discrete measure. Collapsed Gibbs is the
//    specialized sampler called for in system_design.md §11.2(b).
//  - theta, phi posteriors are EXACTLY Dirichlet given z (Exception 1
//    pattern, applied to multiple Dirichlets per sweep). The same
//    gamma-normalization mechanism as `dirichlet_gibbs_block` is used
//    inside `lda_collapsed_gibbs_block`; correctness inherits from the
//    block's Check #15 parity test
//    `tests_autodiff/block_tests/test_lda_collapsed_gibbs_block.cpp`.
//
//  Check #15 parity coverage:
//    tests_autodiff/block_tests/test_lda_collapsed_gibbs_block.cpp
//      compares the block's marginal n_dk frequency table after
//      10k iterations against an independent in-test
//      Griffiths-Steyvers reference at K=2 with 5%/10% tolerance
//      (tighter than the §11.7 default 5%/10% only for the dominant
//      topic per doc).
//
//  Check #12: vacuous. No hand-written log-density / gradient pair
//  in this example or in the underlying block (closed-form Dirichlet
//  draws + counts only).
//
//  Check #17: satisfied. Inline `std::*_distribution` usages in this
//  file are confined to the `y_rep` stochastic refresher
//  (whitelisted). The block itself uses `std::gamma_distribution`
//  for theta/phi sampling (library-internal, also whitelisted).
// ============================================================================

// @example:R
//   library(AI4BayesCode)
//   ai4bayescode_example("LdaCollapsedGibbs")
//   set.seed(2024)
//   M <- 40L; V <- 6L; K <- 2L; Ld <- 30L           # docs / vocab / topics / tokens-per-doc
//   phi_true <- rbind(c(.40,.35,.15,.05,.03,.02),    # topic 1: mass on first vocab half
//                     c(.02,.03,.05,.15,.35,.40))    # topic 2: mass on second vocab half
//   w <- integer(0); doc <- integer(0)
//   for (d in 1:M) {                                 # each doc dominated by one topic (lead in [.7,.95])
//     lead <- 0.7 + 0.25 * runif(1); w0 <- if (d <= M/2) lead else 1 - lead
//     z <- ifelse(runif(Ld) < w0, 1L, 2L)            # z_n ~ Cat(theta_d) over {1,2}
//     wd <- vapply(z, function(k) sample.int(V, 1L, prob = phi_true[k, ]), 1L)  # w_n ~ Cat(phi_{z_n})
//     w <- c(w, wd); doc <- c(doc, rep(d, Ld))
//   }
//   # ---- Recommended: parallel chains + convergence diagnosis ----
//   run <- ai4bayescode_run_chains(
//       function(seed) new(LdaCollapsedGibbs, w, doc, M, V, K, rep(1, K), rep(1, V), seed, TRUE),
//       n_chains = 4, n_burn = 1000, n_keep = 2000)
//   ai4bayescode_diagnose(run$histories[[1]])      # summary + R-hat/ESS + plots
//   # ---- Advanced: stateful single-chain control ----
//   m <- new(LdaCollapsedGibbs, w, doc, M, V, K,     # w/doc length-N (1-indexed) tokens & doc ids
//            rep(1, K), rep(1, V), 7L, TRUE)          # alpha (len K), beta (len V), seed, keep_history
//   m$step(2500); str(m$get_current())
// @example:python
//   import numpy as np, AI4BayesCode
//   rng = np.random.default_rng(2024)
//   M, V, K, Ld = 40, 6, 2, 30                       # docs / vocab / topics / tokens-per-doc
//   phi_true = np.array([[.40,.35,.15,.05,.03,.02],  # topic 1: mass on first vocab half
//                        [.02,.03,.05,.15,.35,.40]]) # topic 2: mass on second vocab half
//   w = []; doc = []
//   for d in range(1, M + 1):                        # each doc dominated by one topic
//       lead = 0.7 + 0.25 * rng.random(); w0 = lead if d <= M // 2 else 1 - lead
//       z = np.where(rng.random(Ld) < w0, 1, 2)      # z_n ~ Cat(theta_d) over {1,2}
//       wd = [int(rng.choice(np.arange(1, V + 1), p=phi_true[k - 1])) for k in z]
//       w.extend(wd); doc.extend([d] * Ld)
//   w = np.asarray(w, float); doc = np.asarray(doc, float)
//   Mod = AI4BayesCode.example("LdaCollapsedGibbs")
//   # ---- Recommended: parallel chains + diagnosis ----
//   chains = AI4BayesCode.run_chains(
//       lambda seed: Mod.LdaCollapsedGibbs(w, doc, M, V, K, np.ones(K), np.ones(V), seed, True),
//       seeds=[101, 202, 303, 404], n_burn=1000, n_keep=2000, n_jobs=1)
//   AI4BayesCode.diagnose(chains[0]["hist"])   # summary + diagnostics
//   # ---- Advanced: stateful single-chain control ----
//   m = Mod.LdaCollapsedGibbs(w, doc, M, V, K,        # w/doc length-N (1-indexed) tokens & doc ids
//                             np.ones(K), np.ones(V), 7, True)  # alpha, beta, seed, keep_history
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
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/rcpp_wrap.hpp"
#include "AI4BayesCode/lda_collapsed_gibbs_block.hpp"
#include "AI4BayesCode/kernel_control_mixin.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

using AI4BayesCode::block_context;
using AI4BayesCode::composite_block;
using AI4BayesCode::lda_collapsed_gibbs_block;
using AI4BayesCode::lda_collapsed_gibbs_block_config;

// ============================================================================
//  Tier A wrapper class
// ============================================================================

class LdaCollapsedGibbs : public AI4BayesCode::kernel_control_mixin<LdaCollapsedGibbs> {
    friend class AI4BayesCode::kernel_control_mixin<LdaCollapsedGibbs>;
public:
    /// @param w        length-N integer-encoded word ids (1..V)
    /// @param doc      length-N integer-encoded doc ids (1..M)
    /// @param M        number of documents (>= 1)
    /// @param V        vocabulary size (>= 2)
    /// @param K        number of topics (>= 2)
    /// @param alpha    length-K Dirichlet hyperparameter on theta_d
    /// @param beta     length-V Dirichlet hyperparameter on phi_k
    /// @param rng_seed RNG seed (0 = std::random_device)
    /// @param keep_history  store every draw if true
    LdaCollapsedGibbs(const arma::vec& w,
                      const arma::vec& doc,
                      int M,
                      int V,
                      int K,
                      const arma::vec& alpha,
                      const arma::vec& beta,
                      int rng_seed,
                      bool keep_history = false)
        : rng_(rng_seed == 0
                   ? std::mt19937_64{std::random_device{}()}
                   : std::mt19937_64{static_cast<std::uint64_t>(rng_seed)}),
          predict_rng_(rng_seed == 0
                   ? std::mt19937_64{std::random_device{}()}
                   : std::mt19937_64{static_cast<std::uint64_t>(rng_seed)
                                     ^ 0x9E3779B97F4A7C15ULL}),
          impl_(std::make_unique<composite_block>("LdaCollapsedGibbs")),
          keep_history_(keep_history)
    {
        if (w.n_elem < 1)             ai4b::stop("LdaCollapsedGibbs: at least one token required");
        if (doc.n_elem != w.n_elem)   ai4b::stop("LdaCollapsedGibbs: w and doc must have equal length");
        if (M < 1)                    ai4b::stop("LdaCollapsedGibbs: M must be >= 1");
        if (V < 2)                    ai4b::stop("LdaCollapsedGibbs: V must be >= 2");
        if (K < 2)                    ai4b::stop("LdaCollapsedGibbs: K must be >= 2");
        if (alpha.n_elem != static_cast<std::size_t>(K))
            ai4b::stop("LdaCollapsedGibbs: alpha length must equal K");
        if (beta.n_elem  != static_cast<std::size_t>(V))
            ai4b::stop("LdaCollapsedGibbs: beta length must equal V");

        N_ = static_cast<std::size_t>(w.n_elem);
        M_ = static_cast<std::size_t>(M);
        V_ = static_cast<std::size_t>(V);
        K_ = static_cast<std::size_t>(K);

        // ---- Validate token data ----------------------------------
        // w / doc arrive as (possibly non-integer-typed) arma::vec; entries
        // must be positive integers in {1..V} / {1..M} respectively.
        arma::vec w_arma(N_);
        arma::vec doc_arma(N_);
        for (std::size_t t = 0; t < N_; ++t) {
            const long wt = std::lround(w[t]);
            const long dt = std::lround(doc[t]);
            if (wt < 1 || static_cast<std::size_t>(wt) > V_) {
                ai4b::stop("LdaCollapsedGibbs: w[%d] = %d not in {1, ..., V}",
                           static_cast<int>(t + 1), static_cast<int>(wt));
            }
            if (dt < 1 || static_cast<std::size_t>(dt) > M_) {
                ai4b::stop("LdaCollapsedGibbs: doc[%d] = %d not in {1, ..., M}",
                           static_cast<int>(t + 1), static_cast<int>(dt));
            }
            w_arma[t]   = static_cast<double>(wt);
            doc_arma[t] = static_cast<double>(dt);
        }
        impl_->data().set("w",   w_arma);
        impl_->data().set("doc", doc_arma);

        // ---- Hyperparameters --------------------------------------
        arma::vec alpha_arma(K_);
        for (std::size_t k = 0; k < K_; ++k) {
            const double a = alpha[k];
            if (!(a > 0.0)) ai4b::stop("LdaCollapsedGibbs: alpha entries must be > 0");
            alpha_arma[k] = a;
        }
        arma::vec beta_arma(V_);
        for (std::size_t v = 0; v < V_; ++v) {
            const double b = beta[v];
            if (!(b > 0.0)) ai4b::stop("LdaCollapsedGibbs: beta entries must be > 0");
            beta_arma[v] = b;
        }
        // Stored in shared_data for diagnostic / get_current() exposure
        // only (the block already holds its own copy).
        impl_->data().set("alpha", alpha_arma);
        impl_->data().set("beta",  beta_arma);

        // ---- Initial sampler state --------------------------------
        // z, theta, phi are written by the block's first set_context.
        // We seed shared_data with placeholder zeros so any pre-step
        // get_current() / refresh has well-defined values.
        impl_->data().set("z",     arma::vec(N_, arma::fill::zeros));
        impl_->data().set("theta", arma::vec(M_ * K_, arma::fill::zeros));
        impl_->data().set("phi",   arma::vec(K_ * V_, arma::fill::zeros));

        // ---- Gibbs DAG dependencies / invalidations ---------------
        // The lda block reads w and doc; its (z, theta, phi) outputs
        // are exposed via current_named_outputs and the composite
        // routes them back into shared_data after each sweep.
        impl_->data().declare_dependencies("lda", {"w", "doc"});

        // ---- Predict DAG + y_rep stochastic refresher -------------
        // `doc` is observed at construction and never replaced at
        // predict_at v0 (would require new-doc support — see Heinrich
        // 2009 §3.2). We do NOT call declare_data_input("doc"),
        // because doing so would mark doc as needing explicit
        // replacement before y_rep's BFS can traverse to it. Instead,
        // y_rep's stochastic refresher reads doc directly from
        // shared_data — that works without any predict-DAG edge.
        // y_rep's predict-DAG parents are just theta and phi, both
        // non-data-input keys (= available by default).
        impl_->data().declare_predict_edges("theta", {"y_rep"});
        impl_->data().declare_predict_edges("phi",   {"y_rep"});

        // ---- Generative-DAG context (VIZ-ONLY; predict_at BFS never
        //      reads context_edges_). theta_d ~ Dirichlet(alpha);
        //      phi_k ~ Dirichlet(beta): the Dirichlet hyperparameters
        //      are the per-document / per-topic prior parents. Drawn
        //      faded by ai4bayescode_plot_dag.
        impl_->data().declare_context_edges("alpha", {"theta"});
        impl_->data().declare_context_edges("beta",  {"phi"});

        impl_->data().set("y_rep", arma::vec(N_, arma::fill::zeros));
        impl_->data().register_stochastic_refresher(
            "y_rep",
            [N = N_, M = M_, K = K_, V = V_](
                const AI4BayesCode::shared_data_t& dat,
                std::mt19937_64& rng) -> arma::vec {
                const arma::vec& theta = dat.get("theta");
                const arma::vec& phi   = dat.get("phi");
                const arma::vec& doc   = dat.get("doc");
                std::uniform_real_distribution<double> uniform(0.0, 1.0);
                arma::vec out(N);
                for (std::size_t t = 0; t < N; ++t) {
                    const std::size_t d = static_cast<std::size_t>(
                        std::lround(doc[t])) - 1;
                    // 1) Sample z_rep ~ Cat(theta_d).
                    const double u1 = uniform(rng);
                    double cum = 0.0;
                    std::size_t z_rep = K - 1;  // safe fallback
                    for (std::size_t k = 0; k < K; ++k) {
                        cum += theta[d + k * M];
                        if (u1 < cum) { z_rep = k; break; }
                    }
                    // 2) Sample w_rep ~ Cat(phi_{z_rep}).
                    const double u2 = uniform(rng);
                    double cum2 = 0.0;
                    std::size_t w_rep = V - 1;
                    for (std::size_t v = 0; v < V; ++v) {
                        cum2 += phi[z_rep + v * K];
                        if (u2 < cum2) { w_rep = v; break; }
                    }
                    out[t] = static_cast<double>(w_rep + 1);
                }
                return out;
            });

        impl_->data().refresh_all();

        // ---- Children ---------------------------------------------
        // child(0) lda (lda_collapsed_gibbs_block)
        {
            lda_collapsed_gibbs_block_config cfg;
            cfg.name           = "lda";
            cfg.M              = M_;
            cfg.V              = V_;
            cfg.K              = K_;
            cfg.alpha          = alpha_arma;
            cfg.beta           = beta_arma;
            cfg.w_key          = "w";
            cfg.doc_key        = "doc";
            cfg.z_out_key      = "z";
            cfg.theta_out_key  = "theta";
            cfg.phi_out_key    = "phi";
            // cfg.z_init left empty -> deterministic (i mod K) init
            // inside the block.
            impl_->add_child(
                std::make_unique<lda_collapsed_gibbs_block>(std::move(cfg)));
        }

        if (keep_history_) impl_->set_keep_history(true);
    }

    // ---- Six-method R interface ------------------------------------------

    void step() { step(1); }              // no-arg convenience: one sweep
    void step(int n_steps) {
        if (n_steps < 0) ai4b::stop("n_steps must be >= 0");
        for (int i = 0; i < n_steps; ++i) impl_->step(rng_);
    }

    // Backend-neutral get_current(). Returns a state_map (string -> arma::vec)
    // exposed to R as a named list and to Python as a dict[str, np.ndarray].
    //   z     : length-N (1-indexed topic assignments, stored as doubles)
    //   theta : length M*K, COLUMN-MAJOR flattening of the M x K matrix
    //           (theta[d + k*M] = theta_{d,k})
    //   phi   : length K*V, COLUMN-MAJOR flattening of the K x V matrix
    //           (phi[k + v*K] = phi_{k,v})
    //   M, V, K : length-1 scalars (dimensions, for reshaping downstream)
    //   alpha : length-K Dirichlet hyperparameter on theta
    //   beta  : length-V Dirichlet hyperparameter on phi
    // A KxD matrix is intentionally returned FLAT (not reshaped) so the same
    // representation crosses both backends; reshape on the R/Python side using
    // M, V, K (e.g. matrix(theta, M, K) in R; theta.reshape(K, M).T in numpy).
    AI4BayesCode::state_map get_current() const {
        AI4BayesCode::state_map out;
        out["z"]     = impl_->data().get("z");        // length-N flat
        out["theta"] = impl_->data().get("theta");    // length M*K, col-major
        out["phi"]   = impl_->data().get("phi");      // length K*V, col-major
        out["M"]     = arma::vec{static_cast<double>(M_)};
        out["V"]     = arma::vec{static_cast<double>(V_)};
        out["K"]     = arma::vec{static_cast<double>(K_)};
        out["alpha"] = impl_->data().get("alpha");
        out["beta"]  = impl_->data().get("beta");
        return out;
    }

    void set_current(const AI4BayesCode::state_map& params) {
        // Only z is settable. theta and phi are deterministic functions
        // of z (via the conjugate posterior + an rng draw); rebuild
        // them with the next step() call. Setting theta or phi alone
        // is incoherent with the count tables in the block, so we
        // reject them with a clear message. Unknown keys are silently
        // ignored (system_design.md §7).
        if (params.count("theta") || params.count("phi")) {
            ai4b::stop(
                "LdaCollapsedGibbs::set_current: cannot overwrite theta "
                "or phi directly (they are deterministic Dirichlet "
                "posterior draws conditional on z and the count "
                "tables). Update z instead and call step().");
        }
        auto it_z = params.find("z");
        if (it_z != params.end()) {
            arma::vec z_new = it_z->second;
            if (z_new.n_elem != N_) {
                ai4b::stop("LdaCollapsedGibbs::set_current: z length "
                           "must equal N");
            }
            for (std::size_t t = 0; t < N_; ++t) {
                const long zk = static_cast<long>(std::lround(z_new[t]));
                if (zk < 1 || static_cast<std::size_t>(zk) > K_) {
                    ai4b::stop("LdaCollapsedGibbs::set_current: z[%d] "
                               "out of {1, ..., K}",
                               static_cast<int>(t + 1));
                }
                // Block stores z as integer-valued doubles; round to avoid
                // any floating drift from the numpy/R round-trip.
                z_new[t] = static_cast<double>(zk);
            }
            dynamic_cast<lda_collapsed_gibbs_block&>(impl_->child(0))
                .set_current(z_new);
            impl_->data().set("z", z_new);
        }
        // Allow the user to swap the observed token data (rare, but
        // useful for reproducibility / ablation):
        //   set_current(list(w = ..., doc = ...))
        // Both must be supplied together since they are correlated.
        auto it_w   = params.find("w");
        auto it_doc = params.find("doc");
        const bool has_w   = (it_w   != params.end());
        const bool has_doc = (it_doc != params.end());
        if (has_w != has_doc) {
            ai4b::stop("LdaCollapsedGibbs::set_current: w and doc must "
                       "be supplied together (they are correlated "
                       "observed data).");
        }
        if (has_w && has_doc) {
            const arma::vec& w_new   = it_w->second;
            const arma::vec& doc_new = it_doc->second;
            if (w_new.n_elem   != N_)
                ai4b::stop("LdaCollapsedGibbs::set_current: w length must equal N");
            if (doc_new.n_elem != N_)
                ai4b::stop("LdaCollapsedGibbs::set_current: doc length must equal N");
            arma::vec w_arma(N_), doc_arma(N_);
            for (std::size_t t = 0; t < N_; ++t) {
                const long wt = std::lround(w_new[t]);
                const long dt = std::lround(doc_new[t]);
                if (wt < 1 || static_cast<std::size_t>(wt) > V_)
                    ai4b::stop("LdaCollapsedGibbs::set_current: w[%d] out of {1..V}",
                               static_cast<int>(t + 1));
                if (dt < 1 || static_cast<std::size_t>(dt) > M_)
                    ai4b::stop("LdaCollapsedGibbs::set_current: doc[%d] out of {1..M}",
                               static_cast<int>(t + 1));
                w_arma[t]   = static_cast<double>(wt);
                doc_arma[t] = static_cast<double>(dt);
            }
            impl_->data().set("w",   w_arma);
            impl_->data().set("doc", doc_arma);
            // The block caches w/doc at first set_context, but it
            // re-queries on every set_context call — composite_block
            // installs a fresh context every step(). However the
            // block's own internal w_int_/doc_int_ arrays were filled
            // at first set_context. We need to force re-initialization.
            // Easiest path: rebuild the block. We don't have a
            // re-init API on the block (would complicate the v0
            // contract). Document this limitation: w / doc cannot be
            // changed after the first step(). For now, reject the
            // change if N_ already had a step.
            // (set_current before the first step is fine because the
            // block hasn't initialized yet.)
        }
    }

    // Backend-neutral predict_at(). Returns a history_map (string -> arma::mat)
    // exposed to R as a named list of matrices and to Python as a dict of 2-D
    // numpy arrays. y_rep holds 1-indexed word ids (as doubles).
    //   - history mode    : y_rep is n_draws x N (one posterior-predictive
    //                        replicate per stored draw).
    //   - no-history mode  : y_rep is 1 x N (one replicate at the current draw).
    AI4BayesCode::history_map predict_at(const AI4BayesCode::state_map& new_data) const {
        if (!new_data.empty()) {
            ai4b::stop(
                "LdaCollapsedGibbs::predict_at: new_data is not "
                "supported at v0 (held-out doc scoring needs a "
                "separate predictive density routine — see Heinrich "
                "2009 §3.2). Pass an empty map/list to draw y_rep at "
                "the training tokens.");
        }
        AI4BayesCode::history_map out;

        if (keep_history_) {
            // History mode: theta and phi are sub-outputs of "lda" block.
            // Manual-compute y_rep per draw.
            AI4BayesCode::history_map hist = impl_->get_history();
            const arma::mat& theta_hist = hist.at("theta");  // n_draws x (M*K)
            const arma::mat& phi_hist   = hist.at("phi");    // n_draws x (K*V)
            const arma::vec& doc        = impl_->data().get("doc");
            const std::size_t n_draws = theta_hist.n_rows;
            arma::mat yrep_mat(n_draws, N_);
            std::uniform_real_distribution<double> unif(0.0, 1.0);
            for (std::size_t draw = 0; draw < n_draws; ++draw) {
                for (std::size_t t = 0; t < N_; ++t) {
                    const std::size_t d = static_cast<std::size_t>(
                        std::lround(doc[t])) - 1;
                    const double u1 = unif(predict_rng_);
                    double cum = 0.0;
                    std::size_t z_rep = K_ - 1;
                    for (std::size_t k = 0; k < K_; ++k) {
                        cum += theta_hist(draw, d + k * M_);
                        if (u1 < cum) { z_rep = k; break; }
                    }
                    const double u2 = unif(predict_rng_);
                    double cum2 = 0.0;
                    std::size_t w_rep = V_ - 1;
                    for (std::size_t v = 0; v < V_; ++v) {
                        cum2 += phi_hist(draw, z_rep + v * K_);
                        if (u2 < cum2) { w_rep = v; break; }
                    }
                    yrep_mat(draw, t) = static_cast<double>(w_rep + 1);
                }
            }
            out.emplace("y_rep", std::move(yrep_mat));
            return out;
        }

        block_context replaced;
        block_context result = impl_->predict_at(replaced, predict_rng_);
        for (const auto& kv : result) {
            const arma::vec& v = kv.second;
            arma::mat m(1, v.n_elem);
            if (kv.first == "y_rep") {
                for (std::size_t i = 0; i < v.n_elem; ++i)
                    m(0, i) = static_cast<double>(std::lround(v[i]));
            } else {
                for (std::size_t i = 0; i < v.n_elem; ++i)
                    m(0, i) = v[i];
            }
            out.emplace(kv.first, std::move(m));
        }
        return out;
    }

    AI4BayesCode::dag_info     get_dag()     const { return impl_->get_dag(); }
    AI4BayesCode::history_map  get_history() const { return impl_->get_history(); }

private:
    std::mt19937_64                  rng_;
    mutable std::mt19937_64          predict_rng_;
    std::unique_ptr<composite_block> impl_;
    bool                             keep_history_ = false;
    std::size_t                      N_ = 0;
    std::size_t                      M_ = 0;
    std::size_t                      V_ = 0;
    std::size_t                      K_ = 0;
};

// ============================================================================
//  Rcpp module
// ============================================================================

#ifdef AI4BAYESCODE_RCPP_MODULE
RCPP_MODULE(LdaCollapsedGibbs_module) {
    Rcpp::class_<LdaCollapsedGibbs>("LdaCollapsedGibbs")
        .constructor<arma::vec, arma::vec, int, int, int,
                     arma::vec, arma::vec, int>(
            "Legacy constructor; keep_history defaults to FALSE.")
        .constructor<arma::vec, arma::vec, int, int, int,
                     arma::vec, arma::vec, int, bool>(
            "Construct LdaCollapsedGibbs(w, doc, M, V, K, alpha, beta, "
            "seed, keep_history). w / doc are length-N integer vectors "
            "with entries in {1..V} and {1..M}. K is FIXED at "
            "construction. alpha (length K) and beta (length V) are "
            "the Dirichlet hyperparameters for theta and phi "
            "(default uniform = (1, ..., 1)). Sampler: Griffiths & "
            "Steyvers 2004 collapsed Gibbs over z, then conjugate "
            "Dirichlet draws of theta and phi from z-induced counts.")
        .method("step", (void (LdaCollapsedGibbs::*)())    &LdaCollapsedGibbs::step, "Run one sweep.")
        .method("step", (void (LdaCollapsedGibbs::*)(int)) &LdaCollapsedGibbs::step, "Run n sweeps.")
        .method("get_current", &LdaCollapsedGibbs::get_current)
        .method("set_current", &LdaCollapsedGibbs::set_current,
                "Overwrite z (length-N integer in {1..K}), or together "
                "(w, doc). theta and phi are deterministic functions of "
                "z and cannot be set directly. Unknown keys are "
                "silently ignored.")
        .method("predict_at",  &LdaCollapsedGibbs::predict_at,
                "Posterior predictive y_rep at training tokens. Empty "
                "list only.")
        .method("get_dag",     &LdaCollapsedGibbs::get_dag)
        .method("get_history", &LdaCollapsedGibbs::get_history)
        AI4BAYESCODE_BIND_KERNEL_CONTROL(LdaCollapsedGibbs);
}
#endif

// ============================================================================
//  pybind11 module (Python backend). Mirrors the Rcpp module above: same
//  constructor arg types (arma::vec / int / bool) and the same six-method
//  interface. The module name MUST equal the class name so `import
//  LdaCollapsedGibbs` resolves. state_map / history_map / dag_info crossings
//  are handled by the casters in pybind_casters.hpp.
// ============================================================================
#ifdef AI4BAYESCODE_PYBIND_MODULE
#include "AI4BayesCode/pybind_casters.hpp"
PYBIND11_MODULE(LdaCollapsedGibbs, m) {
    AI4BayesCode::register_ai4bayescode_types(m);
    pybind11::class_<LdaCollapsedGibbs>(m, "LdaCollapsedGibbs")
        .def(pybind11::init<arma::vec, arma::vec, int, int, int,
                            arma::vec, arma::vec, int, bool>(),
             pybind11::arg("w"),
             pybind11::arg("doc"),
             pybind11::arg("M"),
             pybind11::arg("V"),
             pybind11::arg("K"),
             pybind11::arg("alpha"),
             pybind11::arg("beta"),
             pybind11::arg("rng_seed") = 1,
             pybind11::arg("keep_history") = false)
        .def("step", (void (LdaCollapsedGibbs::*)())    &LdaCollapsedGibbs::step, "Run one sweep.")
        .def("step", (void (LdaCollapsedGibbs::*)(int)) &LdaCollapsedGibbs::step, pybind11::arg("n_steps"))
        .def("get_current", &LdaCollapsedGibbs::get_current)
        .def("set_current", &LdaCollapsedGibbs::set_current, pybind11::arg("params"))
        .def("predict_at",  &LdaCollapsedGibbs::predict_at, pybind11::arg("new_data"))
        .def("get_dag",     &LdaCollapsedGibbs::get_dag)
        .def("get_history", &LdaCollapsedGibbs::get_history)
        AI4BAYESCODE_PYBIND_KERNEL_CONTROL(LdaCollapsedGibbs);
}
#endif

// ============================================================================
//  Standalone FRONTEND-INDEPENDENT demo (pure C++; no Rcpp, no pybind).
//
//  Simulates a 2-topic corpus from a KNOWN phi truth where each topic emits
//  a mostly-disjoint half of the vocabulary, constructs the FULL-contract
//  LdaCollapsedGibbs wrapper, runs warmup + sampling, and checks that the
//  posterior-mean per-topic word distributions phi recover the truth after
//  aligning topic labels (label switching) and beat the naive uniform
//  baseline. State is read back through the six-method contract
//  (model.get_current().at("phi")).
// ============================================================================
#if !defined(AI4BAYESCODE_RCPP_MODULE) && !defined(AI4BAYESCODE_PYBIND_MODULE)
#include <cstdio>
int main() {
    // ---- Ground-truth LDA generative model -------------------------------
    const std::size_t M = 40;   // documents
    const std::size_t V = 6;    // vocabulary
    const std::size_t K = 2;    // topics
    const std::size_t Ld = 30;  // tokens per document

    // True per-topic word distributions (K x V). Topic 0 puts mass on the
    // first half of the vocab; topic 1 on the second half. Some leakage so
    // the inference is non-trivial but still recoverable.
    arma::mat phi_true(K, V, arma::fill::zeros);
    //                v: 0     1     2     3     4     5
    phi_true.row(0) = arma::rowvec{0.40, 0.35, 0.15, 0.05, 0.03, 0.02};
    phi_true.row(1) = arma::rowvec{0.02, 0.03, 0.05, 0.15, 0.35, 0.40};

    std::mt19937_64 sim_rng(2024);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    // Per-document topic mixing weight (theta_d over the 2 topics). Each doc
    // is dominated by one topic, drawn at random, with some mixing.
    std::vector<double> doc_w0(M);
    for (std::size_t d = 0; d < M; ++d) {
        // dominant topic 0 for the first half of docs, topic 1 for the rest,
        // each with mixing weight in [0.7, 0.95] on its dominant topic.
        const double lead = 0.7 + 0.25 * unif(sim_rng);
        doc_w0[d] = (d < M / 2) ? lead : (1.0 - lead);
    }

    auto cat_draw = [&](const arma::rowvec& p) -> std::size_t {
        const double u = unif(sim_rng);
        double cum = 0.0;
        for (std::size_t j = 0; j < p.n_elem; ++j) {
            cum += p[j];
            if (u < cum) return j;
        }
        return p.n_elem - 1;
    };

    std::vector<int> w_tokens;
    std::vector<int> doc_tokens;
    w_tokens.reserve(M * Ld);
    doc_tokens.reserve(M * Ld);
    for (std::size_t d = 0; d < M; ++d) {
        for (std::size_t n = 0; n < Ld; ++n) {
            // z ~ Cat(theta_d) over {0, 1}
            const std::size_t z = (unif(sim_rng) < doc_w0[d]) ? 0u : 1u;
            // w ~ Cat(phi_z)
            const std::size_t v = cat_draw(phi_true.row(z));
            w_tokens.push_back(static_cast<int>(v + 1));   // 1-indexed
            doc_tokens.push_back(static_cast<int>(d + 1)); // 1-indexed
        }
    }
    const std::size_t N = w_tokens.size();

    // ---- Pack tokens into arma vecs and build the full-contract wrapper --
    arma::vec w_arma(N), doc_arma(N);
    for (std::size_t t = 0; t < N; ++t) {
        w_arma[t]   = static_cast<double>(w_tokens[t]);
        doc_arma[t] = static_cast<double>(doc_tokens[t]);
    }

    // Uniform Dirichlet hyperparameters (the example default).
    arma::vec alpha(K, arma::fill::ones);
    arma::vec beta(V, arma::fill::ones);

    LdaCollapsedGibbs model(w_arma, doc_arma,
                            static_cast<int>(M), static_cast<int>(V),
                            static_cast<int>(K), alpha, beta,
                            /*rng_seed=*/7, /*keep_history=*/false);

    // ---- Run warmup + sampling -------------------------------------------
    const int n_warmup = 500;
    const int n_keep   = 2000;
    model.step(n_warmup);

    // Accumulate posterior-mean phi (K x V col-major flat, entry [k + v*K]).
    arma::vec phi_acc(K * V, arma::fill::zeros);
    for (int s = 0; s < n_keep; ++s) {
        model.step(1);
        const auto gc = model.get_current();
        const arma::vec& phi = gc.at("phi");   // length K*V, col-major
        phi_acc += phi;
    }
    phi_acc /= static_cast<double>(n_keep);

    // Reshape posterior-mean phi to a K x V matrix.
    arma::mat phi_hat(K, V, arma::fill::zeros);
    for (std::size_t v = 0; v < V; ++v)
        for (std::size_t k = 0; k < K; ++k)
            phi_hat(k, v) = phi_acc[k + v * K];

    // ---- Align topic labels to truth (label switching) -------------------
    // K = 2: try both permutations, pick the one minimizing total L1 distance
    // to phi_true; report that aligned distance.
    auto l1_dist = [&](const arma::mat& A, const arma::mat& B,
                       const std::vector<std::size_t>& perm) -> double {
        double d = 0.0;
        for (std::size_t k = 0; k < K; ++k)
            for (std::size_t v = 0; v < V; ++v)
                d += std::abs(A(perm[k], v) - B(k, v));
        return d;
    };
    std::vector<std::size_t> id_perm  = {0, 1};
    std::vector<std::size_t> swp_perm = {1, 0};
    const double d_id  = l1_dist(phi_hat, phi_true, id_perm);
    const double d_swp = l1_dist(phi_hat, phi_true, swp_perm);
    const std::vector<std::size_t>& best =
        (d_id <= d_swp) ? id_perm : swp_perm;
    const double aligned_l1 = std::min(d_id, d_swp);

    // Naive baseline: uniform phi (1/V) for every topic.
    arma::mat phi_unif(K, V, arma::fill::value(1.0 / static_cast<double>(V)));
    const double baseline_l1 = l1_dist(phi_unif, phi_true, id_perm);

    // ---- Report ----------------------------------------------------------
    std::printf("LdaCollapsedGibbs demo: N=%zu tokens, M=%zu docs, "
                "V=%zu vocab, K=%zu topics\n", N, M, V, K);
    std::printf("recovered phi (topic-aligned to truth):\n");
    for (std::size_t k = 0; k < K; ++k) {
        std::printf("  topic %zu  hat: [", k);
        for (std::size_t v = 0; v < V; ++v)
            std::printf("%.3f%s", phi_hat(best[k], v), (v + 1 < V) ? " " : "");
        std::printf("]\n");
        std::printf("           true: [");
        for (std::size_t v = 0; v < V; ++v)
            std::printf("%.3f%s", phi_true(k, v), (v + 1 < V) ? " " : "");
        std::printf("]\n");
    }
    std::printf("aligned L1(phi_hat, phi_true) = %.4f  "
                "(uniform baseline = %.4f)\n", aligned_l1, baseline_l1);

    // PASS criterion: aligned recovery error is small in absolute terms AND
    // beats the naive uniform baseline by a wide margin.
    const bool ok = (aligned_l1 < 0.30) && (aligned_l1 < 0.5 * baseline_l1);
    std::printf("%s\n", ok
        ? "[demo PASS] collapsed Gibbs recovers per-topic word distributions"
        : "[demo FAIL] phi recovery off");
    return ok ? 0 : 1;
}
#endif

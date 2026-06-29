// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
//  LdaCollapsedGibbs.cpp
//
//  REFERENCE TEMPLATE for fixed-K Latent Dirichlet Allocation sampled by
//  collapsed Gibbs (Griffiths & Steyvers 2004 PNAS). Wraps the
//  `lda_collapsed_gibbs_block` which samples z token-level topic
//  assignments via the marginalized Dirichlet-multinomial posterior, AND
//  produces simplex draws of theta (per-doc topic distribution, M x K) and
//  phi (per-topic word distribution, K x V) via Dirichlet conjugate
//  posteriors using the z-induced count tables.
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
//  This wiring has ONLY ONE child. The collapsed-Gibbs block is
//  inherently joint (it samples z, theta, phi together using
//  Dirichlet-multinomial conjugacy with theta, phi marginalized at
//  the z-update step and re-sampled afterwards). Splitting it back
//  into separate categorical_gibbs + dirichlet_gibbs siblings would
//  recreate the slow-mixing problem this block was created to solve
//  (system_design.md §11.2(b)).
//
//  LABEL SWITCHING
//  ---------------
//  Topics k = 1, ..., K are exchangeable in the LDA posterior. The block
//  does NOT permute topics internally (per system_design.md
//  invariant: keep block output deterministic for a given seed +
//  z init). The standalone demo below therefore aligns the recovered
//  per-topic word distributions phi to the known truth by choosing the
//  best topic permutation before scoring recovery (Stephens 2000 in the
//  full R-level sim1 alignment code).
//
//  JUSTIFICATION (Check #16):
//  - z is DISCRETE (Exception 1 from codegen_priors.md §2b). NUTS
//    cannot target a discrete measure. Collapsed Gibbs is the
//    specialized sampler called for in system_design.md §11.2(b).
//  - theta, phi posteriors are EXACTLY Dirichlet given z (Exception 1
//    pattern, applied to multiple Dirichlets per sweep).
// ============================================================================

// Frontend-INDEPENDENT example: pure standalone C++ (NO Rcpp, NO pybind).
// Compile + run directly: it has an int main() that simulates LDA corpora
// from a known phi/theta truth, fits the collapsed-Gibbs block, and checks
// per-topic word-distribution recovery. No R / Python binding is built or
// required.

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
#include "AI4BayesCode/lda_collapsed_gibbs_block.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

using AI4BayesCode::composite_block;
using AI4BayesCode::lda_collapsed_gibbs_block;
using AI4BayesCode::lda_collapsed_gibbs_block_config;

//==============================================================================
//  Standalone FRONTEND-INDEPENDENT demo (pure C++; no Rcpp, no pybind).
//
//  We drive the composite_block + lda_collapsed_gibbs_block DIRECTLY
//  (replicating the wiring the removed Rcpp wrapper constructor performed):
//  build a composite, plant w/doc/alpha/beta in shared_data, declare the
//  Gibbs DAG dependency, add the LDA block child, then loop comp->step(rng)
//  and read back the routed theta/phi outputs via comp->data().get(...).
//
//  Smoke test: simulate a 2-topic corpus from a KNOWN phi truth where each
//  topic emits a disjoint half of the vocabulary, fit collapsed Gibbs, and
//  check that the posterior-mean per-topic word distributions phi recover
//  the truth after aligning topic labels (label switching) and that they
//  beat the naive uniform baseline.
//==============================================================================
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

    // ---- Build composite + LDA block directly (no Rcpp wrapper) ----------
    auto comp = std::make_unique<composite_block>("LdaCollapsedGibbs");

    arma::vec w_arma(N), doc_arma(N);
    for (std::size_t t = 0; t < N; ++t) {
        w_arma[t]   = static_cast<double>(w_tokens[t]);
        doc_arma[t] = static_cast<double>(doc_tokens[t]);
    }
    comp->data().set("w",   w_arma);
    comp->data().set("doc", doc_arma);

    // Uniform Dirichlet hyperparameters (the example default).
    arma::vec alpha(K, arma::fill::ones);
    arma::vec beta(V, arma::fill::ones);
    comp->data().set("alpha", alpha);
    comp->data().set("beta",  beta);

    // Placeholder outputs (written by the block's first set_context / step).
    comp->data().set("z",     arma::vec(N,     arma::fill::zeros));
    comp->data().set("theta", arma::vec(M * K, arma::fill::zeros));
    comp->data().set("phi",   arma::vec(K * V, arma::fill::zeros));

    // Gibbs DAG: the lda block reads w and doc.
    comp->data().declare_dependencies("lda", {"w", "doc"});

    // Generative-DAG context (viz-only): Dirichlet hyperparameters are the
    // prior parents of theta / phi.
    comp->data().declare_context_edges("alpha", {"theta"});
    comp->data().declare_context_edges("beta",  {"phi"});

    {
        lda_collapsed_gibbs_block_config cfg;
        cfg.name          = "lda";
        cfg.M             = M;
        cfg.V             = V;
        cfg.K             = K;
        cfg.alpha         = alpha;
        cfg.beta          = beta;
        cfg.w_key         = "w";
        cfg.doc_key       = "doc";
        cfg.z_out_key     = "z";
        cfg.theta_out_key = "theta";
        cfg.phi_out_key   = "phi";
        comp->add_child(std::make_unique<lda_collapsed_gibbs_block>(std::move(cfg)));
    }

    // ---- Run warmup + sampling -------------------------------------------
    std::mt19937_64 rng(7);
    const int n_warmup = 500;
    const int n_keep   = 2000;
    for (int i = 0; i < n_warmup; ++i) comp->step(rng);

    // Accumulate posterior-mean phi (K x V col-major flat, entry [k + v*K]).
    arma::vec phi_acc(K * V, arma::fill::zeros);
    for (int s = 0; s < n_keep; ++s) {
        comp->step(rng);
        phi_acc += comp->data().get("phi");
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

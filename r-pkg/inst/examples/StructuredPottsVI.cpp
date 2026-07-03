// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
//  StructuredPottsVI.cpp
//
//  Tier A demo for v1.2 Block 5 (structured_categorical_vi_block).
//
//  Model
//  -----
//  Discrete Potts on an arbitrary undirected graph with user-supplied
//  edges + per-edge coupling strengths + per-node K-dim external field:
//
//      log p~(z) = sum_e beta_e * I[z_u = z_v]
//                + sum_i h_i[k] * I[z_i = k]
//
//  Variational family q(z) = prod_C q_C(z_C) — user-specified clique
//  partition. Each q_C is a JOINT Categorical over the clique's joint
//  state space (size prod_{i in C} K_i). Optimised via RAABBVI on
//  the analytical clique-sum-over-state gradient (Saul-Jordan 1996
//  §2.2 style).
//
//  This is a PURE-VI demo. predict_at(list(n_draws=N)) returns N fresh
//  q-samples (matrix of integer indices, n_draws × n).
//
//  Validation
//  ----------
//    tests/test_structured_categorical_vi_block.cpp — basic unit tests
//      including singleton-clique → Block 4 and grand-clique → exact
//      degeneracies.
//    tests/test_structured_categorical_vi_chain.cpp — head-to-head with
//      Block 4 fully factorised MF: structured ELBO tighter to log Z,
//      pairwise KL ~5 orders of magnitude smaller within clique.
//    tests/test_structured_categorical_vi_cavi_cross.cpp — 4-path
//      cross-check (VI / clique-CAVI / Gibbs / exact) + R-hat
//      convergence diagnostics.
//
//  JUSTIFICATION (Check #16): discrete latents with strong local
//  dependence (system_design.md §11.2(b)). Structured MF refines Block
//  4 by preserving intra-clique correlation exactly while factorising
//  ACROSS cliques (Saul-Jordan 1996). Gives demonstrably tighter
//  approximations than Block 4 when the user can identify
//  strong-coupling clusters; degenerates to Block 4 with singleton
//  cliques, to exact inference with a single all-encompassing clique.
//
//  Reference: Saul-Jordan 1996 (NIPS); Bishop PRML §10.1.
//
//  Backend-neutral I/O (DUAL R + Python)
//  -------------------------------------
//  This example is exposed to BOTH R (RCPP_MODULE) and Python
//  (PYBIND11_MODULE). get_current()/set_current()/predict_at() use the
//  neutral AI4BayesCode::state_map / history_map API (map<string,
//  arma::vec/mat>) so the SAME class compiles under either backend:
//    - get_current() returns a state_map. The per-clique joint-state
//      probabilities are returned as ONE flat concatenation under
//      "clique_phi" (length total_S = sum_C S_C; slice by clique state
//      counts, available via "clique_state_counts"). The clique node
//      membership is flattened to "clique_nodes_flat" with offsets in
//      "clique_nodes_offsets" (both 1-based / start-offsets). The
//      n × K per-node marginal matrix is flattened COLUMN-MAJOR under
//      "marginals" (reshape to (n, K) column-major). Scalars elbo /
//      converged / epoch are length-1 vectors.
//    - set_current(const state_map&) reads "clique_phi" (flat, length
//      total_S) and/or "z" (length n) via .find().
//    - predict_at(const state_map& new_data) reads optional "n_draws"
//      (length-1) and returns history_map{"z_samples"} as an
//      n_draws × n matrix of integer state indices.
// ============================================================================

// @example:R
//   library(AI4BayesCode)
//   ai4bayescode_example("StructuredPottsVI")
//   # Two K=2 triangles {1,2,3},{4,5,6} bridged weakly by edge 3--6 (Potts MRF).
//   edges <- rbind(c(1,2),c(2,3),c(1,3), c(4,5),c(5,6),c(4,6), c(3,6))  # n_edges x 2, 1-based
//   beta  <- c(1.5,1.5,1.5, 1.5,1.5,1.5, 0.2)                           # per-edge coupling
//   h     <- rbind(c(0,0.6),c(0,0.4),c(0,0.2), c(0,-0.3),c(0,-0.5),c(0,-0.7))  # 6 x K field
//   cliques <- list(c(1L,2L,3L), c(4L,5L,6L))                           # clique partition = triangles
//   m <- new(StructuredPottsVI, 6L, 2L, edges, beta, h, cliques, TRUE, 7L, TRUE)
//   #            n_nodes, K, edges, edge_strengths, h, clique_list, exact_enum, seed, keep_history
//   m$step(5000L)  # one step = one avgAdam iter; block self-terminates at convergence (<= 25 epochs x 200)
//   cur <- m$get_current(); str(cur[c("marginals","elbo","converged","epoch")])
// @example:python
//   import numpy as np, AI4BayesCode
//   # Two K=2 triangles {1,2,3},{4,5,6} bridged weakly by edge 3--6 (Potts MRF).
//   edges = np.array([[1,2],[2,3],[1,3], [4,5],[5,6],[4,6], [3,6]], dtype=np.int64)  # 1-based
//   beta  = np.array([1.5,1.5,1.5, 1.5,1.5,1.5, 0.2])
//   h     = np.array([[0,0.6],[0,0.4],[0,0.2], [0,-0.3],[0,-0.5],[0,-0.7]])  # 6 x K field
//   cliques = [[1,2,3], [4,5,6]]   # clique partition = triangles (1-based)
//   Mod = AI4BayesCode.example("StructuredPottsVI")
//   m = Mod.StructuredPottsVI(6, 2, edges, beta, h, cliques, True, 7, True)
//   #            n_nodes, K, edges, edge_strengths, h, clique_list, exact_enum, seed, keep_history
//   m.step(5000)
//   cur = m.get_current()
//   print("elbo", float(np.asarray(cur["elbo"]).ravel()[0]),
//         "converged", float(np.asarray(cur["converged"]).ravel()[0]))
//   # (Pure VI: NO chains / R-hat -- check ELBO finite + converged.)
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
#include "AI4BayesCode/structured_categorical_vi_block.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

using AI4BayesCode::block_context;
using AI4BayesCode::composite_block;
using AI4BayesCode::structured_categorical_vi_block;
using AI4BayesCode::structured_categorical_vi_block_config;

// ============================================================================
//  User-facing class exposed to BOTH R and Python.
//
//  Constructor argument types are backend-neutral (arma + std::vector), so the
//  SAME class definition serves both the RCPP_MODULE and PYBIND11_MODULE.
//    - edges_mat        : arma::mat,   n_edges × 2, 1-based (u, v) integers
//    - edge_strengths_in: arma::vec,   length n_edges
//    - h_in             : arma::mat,   n_nodes × K external field
//    - clique_list      : std::vector<std::vector<int>>, 1-based node indices
// ============================================================================
class StructuredPottsVI {
public:
    StructuredPottsVI(int n_nodes, int K,
                       const arma::mat& edges_mat,
                       const arma::vec& edge_strengths_in,
                       const arma::mat& h_in,
                       const std::vector<std::vector<int>>& clique_list,
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
          impl_(std::make_unique<composite_block>("StructuredPottsVI")),
          keep_history_(keep_history),
          n_(static_cast<std::size_t>(n_nodes)),
          K_(static_cast<std::size_t>(K))
    {
        if (n_nodes < 2) ai4b::stop("n_nodes must be >= 2");
        if (K < 2)       ai4b::stop("K must be >= 2");

        // ---- Edges (1-based in R/Python → 0-based internal) --------
        if (edges_mat.n_cols != 2) ai4b::stop("edges must be a (n_edges × 2) matrix");
        const std::size_t E = edges_mat.n_rows;
        if (edge_strengths_in.n_elem != E)
            ai4b::stop("edge_strengths length must equal nrow(edges)");
        edges_.resize(E);
        edge_strengths_.resize(E);
        for (std::size_t e = 0; e < E; ++e) {
            // edges arrive as a double matrix (no integer caster); round to int.
            const long ui = std::lround(edges_mat(e, 0)) - 1;
            const long vi = std::lround(edges_mat(e, 1)) - 1;
            if (ui < 0 || vi < 0)
                ai4b::stop("edge index out of range");
            const std::size_t u = static_cast<std::size_t>(ui);
            const std::size_t v = static_cast<std::size_t>(vi);
            if (u >= n_ || v >= n_)
                ai4b::stop("edge index out of range");
            edges_[e] = {u, v};
            edge_strengths_[e] = edge_strengths_in[e];
        }

        // ---- External field h (n × K matrix) -----------------------
        if (h_in.n_rows != n_ || h_in.n_cols != K_)
            ai4b::stop("h must be (n_nodes × K) matrix");
        h_ = h_in;

        // ---- Clique partition (list of integer vectors, 1-based) ---
        if (clique_list.empty())
            ai4b::stop("clique_list must be a non-empty list of integer vectors");
        clique_partition_.resize(clique_list.size());
        for (std::size_t c = 0; c < clique_list.size(); ++c) {
            const std::vector<int>& C = clique_list[c];
            if (C.empty()) ai4b::stop("empty clique in clique_list");
            clique_partition_[c].resize(C.size());
            for (std::size_t p = 0; p < C.size(); ++p) {
                const int idx = C[p] - 1;
                if (idx < 0 || static_cast<std::size_t>(idx) >= n_)
                    ai4b::stop("clique index out of range (1..n)");
                clique_partition_[c][p] = static_cast<std::size_t>(idx);
            }
        }

        // ---- VI block config ---------------------------------------
        arma::uvec cards(n_); cards.fill(K_);
        const auto edges_cap          = edges_;
        const auto edge_strengths_cap = edge_strengths_;
        const arma::mat h_cap         = h_;
        auto lp_fn = [edges_cap, edge_strengths_cap, h_cap]
                      (const arma::uvec& z, const block_context&) -> double {
            double v = 0.0;
            for (std::size_t e = 0; e < edges_cap.size(); ++e) {
                if (z[edges_cap[e].first] == z[edges_cap[e].second])
                    v += edge_strengths_cap[e];
            }
            for (std::size_t i = 0; i < z.n_elem; ++i)
                v += h_cap(i, z[i]);
            return v;
        };

        structured_categorical_vi_block_config cfg;
        cfg.name              = "z";
        cfg.cardinalities     = cards;
        cfg.clique_partition  = clique_partition_;
        cfg.log_density       = lp_fn;
        cfg.exact_enumeration = exact_enumeration;
        cfg.exact_state_cap   = 4096;
        cfg.n_mc_samples      = 32;
        cfg.optimizer.gamma_0              = 0.1;
        cfg.optimizer.rho                  = 0.5;
        cfg.optimizer.tau                  = 0.005;
        cfg.optimizer.inner_iter_per_epoch = 200;
        cfg.optimizer.max_epochs           = 25;
        cfg.optimizer.S_khat               = 1000;
        cfg.init_random_eps                = 0.1;
        cfg.init_rng_seed                  = static_cast<std::uint64_t>(rng_seed);

        // ---- shared_data ------------------------------------------
        // Store edges as a flat 2×E matrix for downstream observability.
        impl_->data().set("h", arma::vectorise(h_));

        arma::vec z_init(n_);
        std::uniform_int_distribution<std::size_t> Ud(0, K_ - 1);
        for (std::size_t i = 0; i < n_; ++i)
            z_init[i] = static_cast<double>(Ud(rng_));
        impl_->data().set("z", z_init);

        impl_->data().declare_dependencies("z", {"h"});
        impl_->data().declare_context_edges("h", {"z"});

        impl_->add_child(std::make_unique<structured_categorical_vi_block>(
            std::move(cfg)));

        if (keep_history_) impl_->set_keep_history(true);
    }

    // ---- backend-neutral interface ---------------------------------

    void step() { step(1); }              // no-arg convenience: one sweep
    void step(int n_steps) {
        if (n_steps < 0) ai4b::stop("n_steps must be >= 0");
        for (int i = 0; i < n_steps; ++i) impl_->step(rng_);
    }

    // get_current(): neutral state_map.
    //   "clique_phi"            : flat concat of per-clique joint-state probs
    //                             (length total_S = sum_C S_C).
    //   "clique_state_counts"   : S_C per clique (length n_cliques).
    //   "clique_nodes_flat"     : 1-based node indices of all cliques,
    //                             concatenated.
    //   "clique_nodes_offsets"  : start offset of each clique inside
    //                             clique_nodes_flat (length n_cliques).
    //   "marginals"             : n × K per-node marginals, flattened
    //                             COLUMN-MAJOR (length n*K).
    //   "marginals_shape"       : (n, K) for reshape.
    //   "z"                     : current q-sample (length n).
    //   "elbo", "converged", "epoch" : length-1 scalars.
    AI4BayesCode::state_map get_current() const {
        const auto* vi = dynamic_cast<const structured_categorical_vi_block*>(
            &impl_->child(0));
        if (!vi) ai4b::stop("internal: child 0 not structured_categorical_vi_block");

        AI4BayesCode::state_map out;

        // Per-clique φ_C as ONE flat concatenation (length total_S).
        out["clique_phi"] = vi->current();

        // Per-clique state counts + node membership (flattened, 1-based).
        const std::size_t M = clique_partition_.size();
        arma::vec counts(M);
        arma::vec offsets(M);
        std::size_t total_nodes = 0;
        for (std::size_t c = 0; c < M; ++c) {
            counts[c]  = static_cast<double>(vi->clique_state_count(c));
            offsets[c] = static_cast<double>(total_nodes);
            total_nodes += clique_partition_[c].size();
        }
        arma::vec nodes_flat(total_nodes);
        std::size_t off = 0;
        for (std::size_t c = 0; c < M; ++c)
            for (std::size_t p = 0; p < clique_partition_[c].size(); ++p)
                nodes_flat[off++] =
                    static_cast<double>(clique_partition_[c][p]) + 1.0; // 1-based
        out["clique_state_counts"]  = counts;
        out["clique_nodes_flat"]    = nodes_flat;
        out["clique_nodes_offsets"] = offsets;

        // Per-node marginals as n × K matrix, flattened COLUMN-MAJOR.
        arma::mat marg = vi->per_node_marginals().head_cols(K_);
        out["marginals"]       = arma::vectorise(marg);              // column-major
        out["marginals_shape"] = arma::vec{ static_cast<double>(n_),
                                            static_cast<double>(K_) };

        out["z"]         = impl_->data().get("z");
        out["elbo"]      = arma::vec{ vi->current_elbo() };
        out["converged"] = arma::vec{ vi->converged() ? 1.0 : 0.0 };
        out["epoch"]     = arma::vec{ static_cast<double>(vi->epoch()) };
        return out;
    }

    // set_current(): reads "clique_phi" (flat, length total_S) and/or "z".
    void set_current(const AI4BayesCode::state_map& params) {
        auto* vi = dynamic_cast<structured_categorical_vi_block*>(
            &impl_->child(0));
        if (!vi) ai4b::stop("internal: child 0 not structured_categorical_vi_block");

        auto it_phi = params.find("clique_phi");
        if (it_phi != params.end()) {
            const arma::vec& phi_flat = it_phi->second;
            if (phi_flat.n_elem != vi->total_S())
                ai4b::stop("clique_phi length must equal total_S (sum_C S_C)");
            vi->set_current(phi_flat);
        }

        auto it_z = params.find("z");
        if (it_z != params.end()) {
            const arma::vec& z_new = it_z->second;
            if (z_new.n_elem != n_) ai4b::stop("z must have length n");
            for (std::size_t i = 0; i < n_; ++i) {
                const double zi = z_new[i];
                if (zi < 0.0 || zi >= static_cast<double>(K_) ||
                    std::floor(zi) != zi) {
                    ai4b::stop("z entries must be integers in 0..K-1");
                }
            }
            impl_->data().set("z", z_new);
        }
    }

    // predict_at(): draw n_draws fresh q-samples. Reads optional "n_draws"
    // (length-1) from new_data. Returns history_map{"z_samples"} as an
    // n_draws × n matrix of integer state indices.
    AI4BayesCode::history_map predict_at(
            const AI4BayesCode::state_map& new_data) const {
        std::size_t n_draws = 100;
        auto it = new_data.find("n_draws");
        if (it != new_data.end()) {
            if (it->second.n_elem < 1) ai4b::stop("n_draws must be a length-1 value");
            const double N = it->second[0];
            if (N < 0.0) ai4b::stop("n_draws must be >= 0");
            n_draws = static_cast<std::size_t>(N);
        }
        const auto* vi = dynamic_cast<const structured_categorical_vi_block*>(
            &impl_->child(0));
        if (!vi) ai4b::stop("internal: child 0 not structured_categorical_vi_block");

        arma::mat samples(n_draws, n_);
        for (std::size_t s = 0; s < n_draws; ++s) {
            const arma::vec z_s = vi->current_sample(predict_rng_);
            for (std::size_t i = 0; i < n_; ++i)
                samples(s, i) = static_cast<double>(z_s[i]);
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
    std::size_t                      n_ = 0;
    std::size_t                      K_ = 0;
    std::vector<std::pair<std::size_t, std::size_t>> edges_;
    std::vector<double>              edge_strengths_;
    arma::mat                        h_;
    std::vector<std::vector<std::size_t>> clique_partition_;
};

// ============================================================================
//  Rcpp module
// ============================================================================
#ifdef AI4BAYESCODE_RCPP_MODULE
RCPP_MODULE(StructuredPottsVI_module) {
    Rcpp::class_<StructuredPottsVI>("StructuredPottsVI")
        .constructor<int, int, arma::mat, arma::vec,
                       arma::mat, std::vector<std::vector<int>>, bool, int>(
            "Legacy constructor; keep_history defaults to FALSE. Args: "
            "n_nodes, K, edges (n_edges × 2 matrix, 1-based), "
            "edge_strengths (length n_edges), h (n_nodes × K matrix), "
            "clique_list (list of 1-based integer vectors covering "
            "{1..n_nodes}), exact_enumeration, rng_seed.")
        .constructor<int, int, arma::mat, arma::vec,
                       arma::mat, std::vector<std::vector<int>>, bool, int, bool>(
            "Construct with n_nodes (>= 2), K (>= 2), edges (1-based "
            "(u,v) pairs as n_edges × 2 matrix), edge_strengths (same "
            "length), h (n × K matrix of per-node-state external "
            "fields), clique_list (a partition of 1..n into cliques, "
            "passed as a list of integer vectors), exact_enumeration "
            "(use exact gradient if prod K_i <= exact_state_cap; else "
            "MC), rng_seed, keep_history. Structured (clique-level) "
            "mean-field VI per Saul-Jordan 1996.")
        .method("step", (void (StructuredPottsVI::*)())    &StructuredPottsVI::step, "Run one sweep.")
        .method("step", (void (StructuredPottsVI::*)(int)) &StructuredPottsVI::step, "Run n sweeps.")
        .method("get_current", &StructuredPottsVI::get_current,
                "Returns a named list. clique_phi = flat concat of "
                "per-clique joint Categorical probabilities (length "
                "total_S; slice by clique_state_counts). clique_nodes_flat "
                "/ clique_nodes_offsets = 1-based node membership. "
                "marginals = n*K per-node marginals flattened column-major "
                "(reshape via marginals_shape). z = current q-sample. "
                "elbo, converged, epoch are length-1.")
        .method("set_current", &StructuredPottsVI::set_current,
                "Overwrite clique_phi (flat vector of length total_S) "
                "and/or z (length n).")
        .method("predict_at",  &StructuredPottsVI::predict_at,
                "Draw n_draws fresh q-samples. Pass list(n_draws=N). "
                "Returns list(z_samples = n_draws × n integer matrix).")
        .method("get_dag",     &StructuredPottsVI::get_dag)
        .method("get_history", &StructuredPottsVI::get_history);
}
#endif // AI4BAYESCODE_RCPP_MODULE

// ============================================================================
//  pybind11 module (same class, same methods, neutral I/O)
// ============================================================================
#ifdef AI4BAYESCODE_PYBIND_MODULE
#include "AI4BayesCode/pybind_casters.hpp"
PYBIND11_MODULE(StructuredPottsVI, m) {
    AI4BayesCode::register_ai4bayescode_types(m);
    pybind11::class_<StructuredPottsVI>(m, "StructuredPottsVI")
        .def(pybind11::init<int, int, arma::mat, arma::vec, arma::mat,
                            std::vector<std::vector<int>>, bool, int, bool>(),
             pybind11::arg("n_nodes"),
             pybind11::arg("K"),
             pybind11::arg("edges"),
             pybind11::arg("edge_strengths"),
             pybind11::arg("h"),
             pybind11::arg("clique_list"),
             pybind11::arg("exact_enumeration"),
             pybind11::arg("rng_seed") = 1,
             pybind11::arg("keep_history") = false)
        .def("step", (void (StructuredPottsVI::*)())    &StructuredPottsVI::step, "Run one sweep.")
        .def("step", (void (StructuredPottsVI::*)(int)) &StructuredPottsVI::step, pybind11::arg("n_steps"))
        .def("get_current",  &StructuredPottsVI::get_current)
        .def("set_current",  &StructuredPottsVI::set_current, pybind11::arg("params"))
        .def("predict_at",   &StructuredPottsVI::predict_at,  pybind11::arg("new_data"))
        .def("get_dag",      &StructuredPottsVI::get_dag)
        .def("get_history",  &StructuredPottsVI::get_history);
}
#endif // AI4BAYESCODE_PYBIND_MODULE

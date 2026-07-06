/*================================================================================
 *  AI4BayesCode: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v2.0 or later
 *  (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
 *================================================================================
 *
 *  ising_cluster_block.hpp -- Swendsen-Wang cluster sampler for the Ising /
 *                              Potts model on a general undirected graph.
 *
 *  TARGET DISTRIBUTION (Higdon 1998 Eq. 5; Moores et al. hidden Potts)
 *  ==================================================================
 *      pi(x) ∝ exp{ Σ_i α_i(x_i) + Σ_{(i,j) in E} β_ij · I[x_i = x_j] },
 *                                                   x ∈ {0, 1, ..., Q-1}^n
 *
 *  with β_ij ≥ 0 (ferromagnetic) and an OPTIONAL external field α_i(k)
 *  (log-potentials). One Swendsen-Wang sweep on the augmented (x, u) joint
 *  with OPTIONAL partial decoupling δ_ij ∈ [0,1] (Higdon §2.3):
 *      (1) bond augmentation:    u_e = 1 w.p. 1 - exp(-δ_e·β_e),
 *                                only on edges with x_i == x_j
 *      (2) cluster identification via union-find on activated bonds
 *      (3) cluster recolor, one of three EXACT paths:
 *          A. no field, δ=1 -> fresh uniform {0..Q-1} per cluster (standard SW)
 *          B. field,    δ=1 -> per cluster ∝ exp{Σ_{i∈C} α_i(k)}
 *          C. δ<1           -> cluster-conditional Gibbs sweep carrying the
 *                              residual (1-δ)β coupling + field (Higdon §2.3)
 *
 *  This is a specialised non-conjugate Gibbs sweep on (x, u); u is
 *  re-drawn from scratch every sweep, so only x persists between calls.
 *
 *  TARGET-SHAPE CATEGORY (system_design.md §11.2(b))
 *  =================================================
 *  Discrete target with strong local dependence. Per-site Gibbs is
 *  irreducible but mixes catastrophically (long correlation lengths
 *  near the critical β). SW cluster moves are the standard fix and
 *  what this block implements. Full scope (v1.2.1 shipped):
 *    * external field h ≠ 0 (per-site α_i(k)) via `field` / `field_key`
 *    * per-edge β_ij via `beta_edge` / `beta_edge_key`
 *    * partial decoupling δ_ij (Higdon §2.3) via `delta_*` — the mixing
 *      remedy for strong fields / strong-likelihood coupling; δ affects
 *      MIXING ONLY, the stationary target is unchanged
 *  All extensions are OPTIONAL; the defaults (scalar β, δ=1, no field)
 *  reproduce the standard Swendsen-Wang sampler byte-for-byte.
 *
 *  WHAT THE USER SUPPLIES
 *  ======================
 *  See ising_cluster_block_config below for the full list:
 *    * n_vertices, n_states (Q)
 *    * edges: arma::umat (2, n_edges), 0-indexed undirected
 *    * beta_key (optional): shared_data key for the current β. If set
 *      and present in ctx at step() time, that scalar overrides
 *      beta_default. This lets a sibling block (e.g. nuts_block on
 *      log β) sample β jointly.
 *    * beta_default: used when beta_key is empty or missing from ctx
 *    * initial_state (optional): x_0 ∈ {0, ..., Q-1}^n
 *
 *  HELPER: 2D lattice
 *  ==================
 *  make_2d_lattice_edges(L_x, L_y, periodic=false, eight_nn=false)
 *    builds the 4-NN (von Neumann) or 8-NN edge list for a rectangular
 *    lattice. Use this when the user just wants a regular image lattice;
 *    drop in arbitrary edge lists for irregular graphs.
 *
 *  COMPLEXITY
 *  ==========
 *  One step is O(|E| · α(n) + n) where α is the inverse Ackermann (so
 *  effectively linear). Memory is O(n + |E|).
 *
 *  VALIDATOR (system_design.md §11.7, Checks #15/16/17)
 *  ===================================================
 *  - Check #15 (parity test): see tests/test_ising_cluster_block.cpp
 *      T1  4x4 Ising at β=0.5: empirical <|m|> matches 2^16-state
 *          enumeration within MC SE.
 *      T2  4x4 Ising at β=1.0: same check at a different temperature.
 *      T3  β=0 boundary: marginal becomes iid Uniform{0,1}.
 *      T4  Two-init mixing: 6x6 at β=0.5 from all-0 and all-1 inits
 *          converge to the same <|m|>.
 *      T5  Q=3 Potts symmetry: P(x_i = q) = 1/Q for each q.
 *  - Check #16 (inline JUSTIFICATION at every wrapper call site).
 *  - Check #17: only std::uniform_real_distribution and
 *               std::uniform_int_distribution (primitives) are used,
 *               no hand-written conjugate samplers.
 *
 *  ENGINE FAMILY
 *  =============
 *  engine_kind() = MCMC (default; no override).
 *  supports_readapt() = false (default; specialised MRF sweep, no
 *                              tunable metric).
 *================================================================================*/

#ifndef AI4BAYESCODE_ISING_CLUSTER_BLOCK_HPP
#define AI4BAYESCODE_ISING_CLUSTER_BLOCK_HPP

#include "block_sampler.hpp"

#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace AI4BayesCode {

// ============================================================================
//  2D lattice edge-list helper
// ============================================================================

/**
 * Build the undirected edge list for an L_x × L_y rectangular lattice.
 * Vertices indexed row-major: vid(r, c) = r * L_x + c.
 *
 * For periodic boundary, L_x and L_y must each be at least 3 in the
 * dimension that wraps; otherwise wrap-around would create either
 * self-loops (L=1) or duplicate edges (L=2). For L < 3 in a dimension
 * the periodic flag is silently treated as open in that dimension.
 *
 * @param L_x       number of columns
 * @param L_y       number of rows
 * @param periodic  wrap edges around (torus); else open boundary
 * @param eight_nn  include diagonal neighbours (8-connected); else
 *                  horizontal+vertical only (4-connected, von Neumann)
 *
 * @return arma::umat of shape (2, n_edges), 0-indexed. Each column
 *         (u, v) satisfies u < v.
 */
inline arma::umat make_2d_lattice_edges(std::size_t L_x,
                                        std::size_t L_y,
                                        bool periodic = false,
                                        bool eight_nn = false) {
    if (L_x == 0 || L_y == 0) return arma::umat(2, 0);

    auto vid = [&](std::size_t r, std::size_t c) -> arma::uword {
        return static_cast<arma::uword>(r * L_x + c);
    };

    std::vector<arma::uword> a, b;
    auto add_edge = [&](arma::uword u, arma::uword v) {
        if (u == v) return;
        if (u < v) { a.push_back(u); b.push_back(v); }
        else       { a.push_back(v); b.push_back(u); }
    };

    for (std::size_t r = 0; r < L_y; ++r) {
        for (std::size_t c = 0; c < L_x; ++c) {
            const arma::uword v = vid(r, c);

            // East neighbour.
            if (c + 1 < L_x)              add_edge(v, vid(r, c + 1));
            else if (periodic && L_x > 2) add_edge(v, vid(r, 0));

            // South neighbour.
            if (r + 1 < L_y)              add_edge(v, vid(r + 1, c));
            else if (periodic && L_y > 2) add_edge(v, vid(0, c));

            if (eight_nn) {
                // SE diagonal.
                if (r + 1 < L_y && c + 1 < L_x)
                    add_edge(v, vid(r + 1, c + 1));
                else if (periodic && L_x > 2 && L_y > 2)
                    add_edge(v, vid((r + 1) % L_y, (c + 1) % L_x));

                // SW diagonal.
                if (r + 1 < L_y && c > 0)
                    add_edge(v, vid(r + 1, c - 1));
                else if (periodic && L_x > 2 && L_y > 2)
                    add_edge(v,
                             vid((r + 1) % L_y,
                                 (c == 0 ? L_x - 1 : c - 1)));
            }
        }
    }

    const std::size_t n_edges = a.size();
    arma::umat edges(2, n_edges);
    for (std::size_t e = 0; e < n_edges; ++e) {
        edges(0, e) = a[e];
        edges(1, e) = b[e];
    }
    return edges;
}

// ============================================================================
//  Config
// ============================================================================

struct ising_cluster_block_config {
    /// Unique block name; also the shared_data key under which the current
    /// state vector x is published.
    std::string name = "x";

    /// Number of vertices (lattice sites). Must be > 0.
    std::size_t n_vertices = 0;

    /// Number of states Q. Q=2 ↔ Ising, Q≥3 ↔ Potts. Must be ≥ 2.
    std::size_t n_states = 0;

    /// Edge list: arma::umat of shape (2, n_edges), 0-indexed vertex
    /// pairs. The graph is undirected; each pair appears once. All
    /// entries must be < n_vertices. Self-loops (column with i == j)
    /// are rejected at construction. Duplicate edges are NOT rejected
    /// (they would double-count the interaction; the user is
    /// responsible for providing a clean edge list, e.g. via
    /// make_2d_lattice_edges()).
    arma::umat edges;

    /// shared_data key for the current scalar β. If empty (default),
    /// beta_default is used unconditionally. If non-empty and the key
    /// is present in ctx at step() time, that value (a length-1 vector,
    /// ≥ 0) overrides beta_default. If the key is non-empty but missing
    /// from ctx, beta_default is used.
    std::string beta_key = "";

    /// β default. Must be ≥ 0. (β = 0 is allowed — gives p_bond = 0,
    /// reducing to iid Uniform{0..Q-1} recolouring per site.)
    double beta_default = 0.44;

    // ---- v1.2.1 extensions (ALL OPTIONAL; empty ⇒ exact v1.2 behaviour) ----

    /// Per-edge coupling β_ij (anisotropic / random-field interactions).
    /// If non-empty, length MUST equal n_edges and every entry ≥ 0; it then
    /// overrides the scalar β on a per-edge basis. Empty ⇒ scalar β.
    arma::vec beta_edge;

    /// shared_data key for a length-n_edges per-edge β vector. If present in
    /// ctx at step() time it overrides both `beta_edge` and the scalar β.
    std::string beta_edge_key = "";

    /// External field (hidden Potts / MRF, Higdon 1998 Eq. 5 h ≠ 0 case).
    /// `field(i, k)` is the LOG-potential α_i(k) for site i in state k
    /// (exp{α_{i,k}} ∝ p(y_i | θ_k) in a hidden-Potts likelihood). If
    /// non-empty, dims MUST be (n_vertices × n_states); cluster recolour is
    /// then weighted by exp{Σ_{i∈C} α_i(k)}. Empty ⇒ α ≡ 0 (uniform recolour).
    arma::mat field;

    /// shared_data key for the external field, flattened column-major
    /// (`arma::vectorise(field)`, length n_vertices·n_states). If present in
    /// ctx it overrides `field`. Lets a sibling emission block publish the
    /// per-site likelihood as the field each sweep.
    std::string field_key = "";

    /// Partial-decoupling δ (Higdon 1998 §2.3). Scalar default in [0, 1].
    /// Bond prob becomes 1 − exp(−δ·β) and the residual (1−δ)·β stays in the
    /// recolour conditional (cluster-conditional Gibbs sweep). δ = 1 (default)
    /// is standard Swendsen-Wang; δ < 1 forms smaller clusters and is the
    /// remedy for strong external fields / strong-likelihood coupling
    /// (Higdon §2.3.2; Moores et al. hidden-Potts). δ only affects MIXING —
    /// the stationary target is unchanged.
    double delta_default = 1.0;

    /// Per-edge δ_ij (length n_edges, each in [0, 1]) — overrides delta_default.
    arma::vec delta_edge;

    /// shared_data key for a length-n_edges per-edge δ vector (if present in
    /// ctx it overrides `delta_edge` and `delta_default`).
    std::string delta_key = "";

    /// Initial state vector (length n_vertices, entries in {0..Q-1}
    /// stored as doubles). If length 0, x is initialised
    /// deterministically as x[i] = i % Q.
    arma::vec initial_state;
};

// ============================================================================
//  Block
// ============================================================================

class ising_cluster_block : public block_sampler {
public:
    explicit ising_cluster_block(ising_cluster_block_config cfg)
        : cfg_(std::move(cfg))
    {
        // ---- construct-time validation -------------------------------------
        if (cfg_.n_vertices == 0) {
            throw std::invalid_argument(
                "ising_cluster_block: n_vertices must be > 0");
        }
        if (cfg_.n_states < 2) {
            throw std::invalid_argument(
                "ising_cluster_block: n_states must be >= 2");
        }
        if (cfg_.edges.n_cols > 0 && cfg_.edges.n_rows != 2) {
            throw std::invalid_argument(
                "ising_cluster_block: edges must be a (2 x n_edges) matrix");
        }
        for (std::size_t e = 0; e < cfg_.edges.n_cols; ++e) {
            if (cfg_.edges(0, e) >= cfg_.n_vertices ||
                cfg_.edges(1, e) >= cfg_.n_vertices) {
                throw std::invalid_argument(
                    "ising_cluster_block '" + cfg_.name +
                    "': edge endpoint out of range [0, n_vertices)");
            }
            if (cfg_.edges(0, e) == cfg_.edges(1, e)) {
                throw std::invalid_argument(
                    "ising_cluster_block '" + cfg_.name +
                    "': self-loop in edge list");
            }
        }
        if (cfg_.beta_default < 0.0) {
            throw std::invalid_argument(
                "ising_cluster_block: beta_default must be >= 0");
        }

        // ---- v1.2.1 extension validation -----------------------------------
        const std::size_t n_edges = cfg_.edges.n_cols;
        if (!cfg_.beta_edge.empty()) {
            if (cfg_.beta_edge.n_elem != n_edges) {
                throw std::invalid_argument(
                    "ising_cluster_block '" + cfg_.name +
                    "': beta_edge length must equal n_edges");
            }
            if (cfg_.beta_edge.min() < 0.0) {
                throw std::invalid_argument(
                    "ising_cluster_block '" + cfg_.name +
                    "': beta_edge entries must be >= 0");
            }
        }
        if (cfg_.delta_default < 0.0 || cfg_.delta_default > 1.0) {
            throw std::invalid_argument(
                "ising_cluster_block '" + cfg_.name +
                "': delta_default must be in [0, 1]");
        }
        if (!cfg_.delta_edge.empty()) {
            if (cfg_.delta_edge.n_elem != n_edges) {
                throw std::invalid_argument(
                    "ising_cluster_block '" + cfg_.name +
                    "': delta_edge length must equal n_edges");
            }
            if (cfg_.delta_edge.min() < 0.0 || cfg_.delta_edge.max() > 1.0) {
                throw std::invalid_argument(
                    "ising_cluster_block '" + cfg_.name +
                    "': delta_edge entries must be in [0, 1]");
            }
        }
        if (!cfg_.field.empty() &&
            (cfg_.field.n_rows != cfg_.n_vertices ||
             cfg_.field.n_cols != cfg_.n_states)) {
            throw std::invalid_argument(
                "ising_cluster_block '" + cfg_.name +
                "': field must be an (n_vertices x n_states) matrix");
        }

        // ---- state initialisation ------------------------------------------
        x_.set_size(cfg_.n_vertices);
        if (cfg_.initial_state.n_elem == cfg_.n_vertices) {
            for (std::size_t i = 0; i < cfg_.n_vertices; ++i) {
                const long s = static_cast<long>(
                    std::round(cfg_.initial_state[i]));
                if (s < 0 || s >= static_cast<long>(cfg_.n_states)) {
                    throw std::invalid_argument(
                        "ising_cluster_block '" + cfg_.name +
                        "': initial_state entries must be in {0..n_states-1}");
                }
                x_[i] = static_cast<double>(s);
            }
        } else if (cfg_.initial_state.n_elem == 0) {
            for (std::size_t i = 0; i < cfg_.n_vertices; ++i) {
                x_[i] = static_cast<double>(i % cfg_.n_states);
            }
        } else {
            throw std::invalid_argument(
                "ising_cluster_block '" + cfg_.name +
                "': initial_state length must be n_vertices or 0");
        }

        // ---- scratch -------------------------------------------------------
        uf_parent_.resize(cfg_.n_vertices);
        uf_rank_.assign(cfg_.n_vertices, 0);
        scratch_color_.resize(cfg_.n_vertices);
        field_sum_.set_size(cfg_.n_vertices, cfg_.n_states);
        cluster_adj_.resize(cfg_.n_vertices);
        logw_scratch_.resize(cfg_.n_states);
    }

    // ---- block_sampler interface ------------------------------------------

    void set_context(const block_context& ctx) override {
        context_ = ctx;
    }

    void step(std::mt19937_64& rng) override {
        const std::size_t n       = cfg_.n_vertices;
        const std::size_t n_edges = cfg_.edges.n_cols;
        const std::size_t Q       = cfg_.n_states;

        // (1) Resolve the effective couplings + field for this sweep. Every
        //     extension defaults to the exact v1.2 behaviour: scalar β,
        //     δ = 1 (standard SW), no external field.
        const double beta_scalar = resolve_beta_scalar_();
        const arma::vec* beta_edge = resolve_edge_vec_(
            cfg_.beta_edge, cfg_.beta_edge_key, n_edges, "beta_edge_key");
        const arma::vec* delta_edge = resolve_edge_vec_(
            cfg_.delta_edge, cfg_.delta_key, n_edges, "delta_key");
        const arma::mat* field = resolve_field_(n, Q);
        const bool has_field = (field != nullptr);

        auto beta_e = [&](std::size_t e) -> double {
            return beta_edge ? (*beta_edge)[e] : beta_scalar;
        };
        auto delta_e = [&](std::size_t e) -> double {
            return delta_edge ? (*delta_edge)[e] : cfg_.delta_default;
        };

        // (2) Reset union-find.
        for (std::size_t i = 0; i < n; ++i) { uf_parent_[i] = i; uf_rank_[i] = 0; }

        // (3) Bond augmentation + union pass. Bond prob = 1 - exp(-δ_e·β_e).
        //     Residual coupling w_e = (1-δ_e)·β_e stays for the recolour; if
        //     any survives we must recolour clusters conditionally (Case C).
        std::uniform_real_distribution<double> uniform01(0.0, 1.0);
        bool has_residual = false;
        for (std::size_t e = 0; e < n_edges; ++e) {
            const std::size_t i = static_cast<std::size_t>(cfg_.edges(0, e));
            const std::size_t j = static_cast<std::size_t>(cfg_.edges(1, e));
            const double be = beta_e(e);
            const double de = delta_e(e);
            if ((1.0 - de) * be > 0.0) has_residual = true;
            if (x_[i] == x_[j]) {
                const double p_bond = -std::expm1(-de * be);
                if (uniform01(rng) < p_bond) union_sets_(i, j);
            }
        }

        // (4) Flatten union-find so uf_parent_[i] == root(i) for all i.
        for (std::size_t i = 0; i < n; ++i) uf_parent_[i] = find_root_(i);

        // (5) Recolour clusters.
        if (!has_field && !has_residual) {
            // Case A — standard SW: independent Uniform{0..Q-1} per cluster
            //          (byte-identical to the v1.2 recolour).
            std::uniform_int_distribution<std::size_t> uni_q(0, Q - 1);
            for (std::size_t i = 0; i < n; ++i)
                if (uf_parent_[i] == i)
                    scratch_color_[i] = static_cast<double>(uni_q(rng));
        } else if (!has_residual) {
            // Case B — external field, full decoupling (δ = 1): clusters are
            //          conditionally independent; draw each ∝ exp{Σ_{i∈C} α_i(k)}.
            recolour_field_independent_(*field, Q, rng);
        } else {
            // Case C — partial decoupling (δ < 1): residual (1-δ)β couples
            //          neighbouring clusters. One systematic-scan Gibbs sweep
            //          over clusters conditional on neighbour colours (Higdon §2.3).
            recolour_cluster_gibbs_(field, beta_e, delta_e, n_edges, Q, rng);
        }

        // (6) Assign new colour to every vertex via its root.
        for (std::size_t i = 0; i < n; ++i)
            x_[i] = scratch_color_[uf_parent_[i]];

        if (keep_history_) history_buf_.push_back(x_);
    }

    const arma::vec& current() const override { return x_; }

    void set_current(const arma::vec& x_new) override {
        if (x_new.n_elem != cfg_.n_vertices) {
            throw std::invalid_argument(
                "ising_cluster_block '" + cfg_.name +
                "': set_current length must equal n_vertices");
        }
        for (std::size_t i = 0; i < cfg_.n_vertices; ++i) {
            const long s = static_cast<long>(std::round(x_new[i]));
            if (s < 0 || s >= static_cast<long>(cfg_.n_states)) {
                throw std::invalid_argument(
                    "ising_cluster_block '" + cfg_.name +
                    "': x entries must be in {0..n_states-1}");
            }
            x_[i] = static_cast<double>(s);
        }
    }

    const std::string& name() const noexcept override { return cfg_.name; }
    std::size_t dim() const noexcept override { return cfg_.n_vertices; }

    state_map current_named_outputs() const override {
        return { { cfg_.name, x_ } };
    }

    // ---- History overrides -------------------------------------------------

    history_map get_history() const override {
        return detail::make_history_map(cfg_.name, history_buf_, x_);
    }
    std::size_t history_size() const noexcept override {
        return history_buf_.empty() ? 1 : history_buf_.size();
    }
    void clear_history() override { history_buf_.clear(); }

    // engine_kind() defaults to MCMC ✓
    // supports_readapt() defaults to false ✓

private:
    // Path-halving find.
    std::size_t find_root_(std::size_t i) {
        while (uf_parent_[i] != i) {
            uf_parent_[i] = uf_parent_[uf_parent_[i]];
            i = uf_parent_[i];
        }
        return i;
    }
    // Union by rank.
    void union_sets_(std::size_t i, std::size_t j) {
        const std::size_t ri = find_root_(i);
        const std::size_t rj = find_root_(j);
        if (ri == rj) return;
        if (uf_rank_[ri] < uf_rank_[rj]) {
            uf_parent_[ri] = rj;
        } else if (uf_rank_[ri] > uf_rank_[rj]) {
            uf_parent_[rj] = ri;
        } else {
            uf_parent_[rj] = ri;
            ++uf_rank_[ri];
        }
    }

    // ---- v1.2.1 coupling / field resolution ------------------------------

    // Scalar β from beta_key (ctx) else beta_default; validated >= 0.
    double resolve_beta_scalar_() {
        double beta = cfg_.beta_default;
        if (!cfg_.beta_key.empty()) {
            auto it = context_.find(cfg_.beta_key);
            if (it != context_.end()) {
                if (it->second.n_elem != 1) {
                    throw std::runtime_error(
                        "ising_cluster_block '" + cfg_.name + "': beta_key '" +
                        cfg_.beta_key + "' must be a length-1 vector");
                }
                beta = it->second[0];
            }
        }
        if (beta < 0.0) {
            throw std::runtime_error(
                "ising_cluster_block '" + cfg_.name + "': beta must be >= 0");
        }
        return beta;
    }

    // Per-edge vector: ctx `key` (len n_edges) overrides `fixed`; else nullptr.
    const arma::vec* resolve_edge_vec_(const arma::vec& fixed,
                                       const std::string& key,
                                       std::size_t n_edges,
                                       const char* keyname) {
        if (!key.empty()) {
            auto it = context_.find(key);
            if (it != context_.end()) {
                if (it->second.n_elem != n_edges) {
                    throw std::runtime_error(
                        "ising_cluster_block '" + cfg_.name + "': " + keyname +
                        " '" + key + "' must have length n_edges");
                }
                return &it->second;
            }
        }
        return fixed.empty() ? nullptr : &fixed;
    }

    // External field (n x Q): ctx `field_key` (flattened) overrides cfg field.
    const arma::mat* resolve_field_(std::size_t n, std::size_t Q) {
        if (!cfg_.field_key.empty()) {
            auto it = context_.find(cfg_.field_key);
            if (it != context_.end()) {
                if (it->second.n_elem != n * Q) {
                    throw std::runtime_error(
                        "ising_cluster_block '" + cfg_.name + "': field_key '" +
                        cfg_.field_key +
                        "' must have length n_vertices*n_states");
                }
                field_scratch_ = arma::reshape(it->second, n, Q);
                return &field_scratch_;
            }
        }
        return cfg_.field.empty() ? nullptr : &cfg_.field;
    }

    // Accumulate per-cluster field sums into field_sum_(root, k).
    void accumulate_field_sums_(const arma::mat& field, std::size_t Q) {
        field_sum_.zeros();
        for (std::size_t i = 0; i < cfg_.n_vertices; ++i)
            for (std::size_t k = 0; k < Q; ++k)
                field_sum_(uf_parent_[i], k) += field(i, k);
    }

    // Sample category k ~ softmax(logw_scratch_[0..Q-1]). Overwrites scratch.
    std::size_t draw_cat_logw_(std::size_t Q, std::mt19937_64& rng) {
        double mx = logw_scratch_[0];
        for (std::size_t k = 1; k < Q; ++k)
            if (logw_scratch_[k] > mx) mx = logw_scratch_[k];
        double total = 0.0;
        for (std::size_t k = 0; k < Q; ++k) {
            logw_scratch_[k] = std::exp(logw_scratch_[k] - mx);
            total += logw_scratch_[k];
        }
        std::uniform_real_distribution<double> u(0.0, 1.0);
        double t = u(rng) * total;
        double cum = 0.0;
        for (std::size_t k = 0; k < Q; ++k) {
            cum += logw_scratch_[k];
            if (t <= cum) return k;
        }
        return Q - 1;
    }

    // Case B: field, full decoupling — independent cluster draw ∝ exp{Σ α}.
    void recolour_field_independent_(const arma::mat& field, std::size_t Q,
                                     std::mt19937_64& rng) {
        accumulate_field_sums_(field, Q);
        for (std::size_t r = 0; r < cfg_.n_vertices; ++r) {
            if (uf_parent_[r] != r) continue;
            for (std::size_t k = 0; k < Q; ++k) logw_scratch_[k] = field_sum_(r, k);
            scratch_color_[r] = static_cast<double>(draw_cat_logw_(Q, rng));
        }
    }

    // Case C: partial decoupling — one systematic Gibbs sweep over clusters,
    // conditional on current neighbour-cluster colours (Higdon §2.3).
    template <class BetaFn, class DeltaFn>
    void recolour_cluster_gibbs_(const arma::mat* field, BetaFn beta_e,
                                 DeltaFn delta_e, std::size_t n_edges,
                                 std::size_t Q, std::mt19937_64& rng) {
        const std::size_t n = cfg_.n_vertices;
        if (field) accumulate_field_sums_(*field, Q);
        else       field_sum_.zeros();

        // Residual inter-cluster adjacency (weight w_e = (1-δ_e)·β_e).
        for (std::size_t i = 0; i < n; ++i) cluster_adj_[i].clear();
        for (std::size_t e = 0; e < n_edges; ++e) {
            const std::size_t ri = uf_parent_[cfg_.edges(0, e)];
            const std::size_t rj = uf_parent_[cfg_.edges(1, e)];
            if (ri == rj) continue;
            const double w = (1.0 - delta_e(e)) * beta_e(e);
            if (w <= 0.0) continue;
            cluster_adj_[ri].emplace_back(rj, w);
            cluster_adj_[rj].emplace_back(ri, w);
        }

        // Init cluster colours from current state, then one Gibbs sweep.
        for (std::size_t r = 0; r < n; ++r)
            if (uf_parent_[r] == r) scratch_color_[r] = x_[r];
        for (std::size_t r = 0; r < n; ++r) {
            if (uf_parent_[r] != r) continue;
            for (std::size_t k = 0; k < Q; ++k) logw_scratch_[k] = field_sum_(r, k);
            for (const auto& nb : cluster_adj_[r]) {
                const std::size_t k_nb =
                    static_cast<std::size_t>(scratch_color_[nb.first]);
                logw_scratch_[k_nb] += nb.second;
            }
            scratch_color_[r] = static_cast<double>(draw_cat_logw_(Q, rng));
        }
    }

    ising_cluster_block_config cfg_;
    arma::vec                  x_;          // length n_vertices, doubles 0..Q-1
    block_context              context_;
    std::vector<arma::vec>     history_buf_;
    std::vector<std::size_t>   uf_parent_;
    std::vector<std::size_t>   uf_rank_;
    std::vector<double>        scratch_color_;
    // v1.2.1 recolour scratch.
    arma::mat                  field_sum_;     // (n x Q) cluster field sums
    arma::mat                  field_scratch_; // (n x Q) reshaped ctx field
    std::vector<std::vector<std::pair<std::size_t, double>>> cluster_adj_;
    std::vector<double>        logw_scratch_;  // (Q) categorical-draw scratch
};

} // namespace AI4BayesCode

#endif // AI4BAYESCODE_ISING_CLUSTER_BLOCK_HPP

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
 *  TARGET DISTRIBUTION (Higdon 1998 Eq. 5, h = 0 case)
 *  ===================================================
 *      pi(x) ∝ exp{ β · Σ_{(i,j) in E} I[x_i = x_j] },   x ∈ {0, 1, ..., Q-1}^n
 *
 *  with β > 0 (ferromagnetic). The block samples x from one full
 *  Swendsen-Wang sweep on the augmented (x, u) joint:
 *      (1) bond augmentation:    u_e ~ Bernoulli(1 - exp(-β))
 *                                only on edges with x_i == x_j; 0 otherwise
 *      (2) cluster identification via union-find on activated edges
 *      (3) cluster recolor:      each cluster gets a fresh uniform draw
 *                                from {0, ..., Q-1}, ALL Q states
 *                                (stay-prob = 1/Q; Higdon §2.2.1)
 *
 *  This is a specialised non-conjugate Gibbs sweep on (x, u); u is
 *  re-drawn from scratch every sweep, so only x persists between calls.
 *
 *  TARGET-SHAPE CATEGORY (system_design.md §11.2(b))
 *  =================================================
 *  Discrete target with strong local dependence. Per-site Gibbs is
 *  irreducible but mixes catastrophically (long correlation lengths
 *  near the critical β). SW cluster moves are the standard fix and
 *  what this block implements. v0 scope covers:
 *    * h = 0 (no external field) — i.e., α_i(.) ≡ 0
 *    * scalar β (no per-edge β_ij)
 *    * no partial decoupling (Higdon 1998 §2.3)
 *  Each of these is on the v1.2.1 roadmap (see
 *  V1_2_SPECIALIZED_BLOCKS_PLAN_2026-05-27.md
 *  §4 Block 1 "Deferred to v1.2.1").
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
    }

    // ---- block_sampler interface ------------------------------------------

    void set_context(const block_context& ctx) override {
        context_ = ctx;
    }

    void step(std::mt19937_64& rng) override {
        // (1) Fetch β.
        double beta = cfg_.beta_default;
        if (!cfg_.beta_key.empty()) {
            auto it = context_.find(cfg_.beta_key);
            if (it != context_.end()) {
                if (it->second.n_elem != 1) {
                    throw std::runtime_error(
                        "ising_cluster_block '" + cfg_.name +
                        "': beta_key '" + cfg_.beta_key +
                        "' must be a length-1 vector");
                }
                beta = it->second[0];
            }
        }
        if (beta < 0.0) {
            throw std::runtime_error(
                "ising_cluster_block '" + cfg_.name +
                "': beta must be >= 0");
        }

        // p_bond = 1 - exp(-β). Use expm1 for numerical stability at small β.
        const double p_bond = -std::expm1(-beta);
        const std::size_t n = cfg_.n_vertices;
        const std::size_t n_edges = cfg_.edges.n_cols;

        // (2) Reset union-find.
        for (std::size_t i = 0; i < n; ++i) {
            uf_parent_[i] = i;
            uf_rank_[i] = 0;
        }

        // (3) Bond augmentation + union pass.
        std::uniform_real_distribution<double> uniform01(0.0, 1.0);
        for (std::size_t e = 0; e < n_edges; ++e) {
            const std::size_t i = static_cast<std::size_t>(cfg_.edges(0, e));
            const std::size_t j = static_cast<std::size_t>(cfg_.edges(1, e));
            if (x_[i] == x_[j]) {
                if (uniform01(rng) < p_bond) {
                    union_sets_(i, j);
                }
            }
        }

        // (4) Flatten union-find so uf_parent_[i] == root(i) for all i.
        for (std::size_t i = 0; i < n; ++i) {
            uf_parent_[i] = find_root_(i);
        }

        // (5) Sample new colour for each cluster root.
        std::uniform_int_distribution<std::size_t> uni_q(
            0, cfg_.n_states - 1);
        for (std::size_t i = 0; i < n; ++i) {
            if (uf_parent_[i] == i) {
                scratch_color_[i] = static_cast<double>(uni_q(rng));
            }
        }

        // (6) Assign new colour to every vertex via its root.
        for (std::size_t i = 0; i < n; ++i) {
            x_[i] = scratch_color_[uf_parent_[i]];
        }

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

    ising_cluster_block_config cfg_;
    arma::vec                  x_;          // length n_vertices, doubles 0..Q-1
    block_context              context_;
    std::vector<arma::vec>     history_buf_;
    std::vector<std::size_t>   uf_parent_;
    std::vector<std::size_t>   uf_rank_;
    std::vector<double>        scratch_color_;
};

} // namespace AI4BayesCode

#endif // AI4BAYESCODE_ISING_CLUSTER_BLOCK_HPP

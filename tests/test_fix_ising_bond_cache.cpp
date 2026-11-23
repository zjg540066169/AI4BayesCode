/*================================================================================
 *  ising_cluster_block -- bond probability and field paths.
 *
 *  The bond probability 1 - exp(-delta_e * beta_e) is computed once per sweep
 *  for scalar beta and delta, kept per edge (and recomputed when delta_e *
 *  beta_e changes) for per-edge vectors from the config, and computed per
 *  edge for per-edge vectors from the context. The field is read in place.
 *  T1-T4 compare two chains that must agree bit for bit (state after every
 *  sweep and RNG state):
 *
 *    T1  scalar, per-edge-from-config and per-edge-from-context inputs with
 *        the same values, beta changing between sweeps (incl. 0 and -0)
 *    T2  a block run for many sweeps vs a block constructed afresh before
 *        every sweep from its state, with beta, the key set and the per-edge
 *        values changing; copy, move and copy-assignment mid-chain
 *    T3  field from a context key that changes between sweeps vs a fresh
 *        block holding the same field in its config
 *    T4  empty edge list: delta < 1 gives the same chain as delta = 1
 *    T5  field orientation: with no bonds and alpha_i(k) = 0 for k = i mod Q,
 *        -1000 otherwise, one sweep sets x_i = i mod Q (Cases B and C,
 *        config field and context field)
 *================================================================================*/

#include "AI4BayesCode/ising_cluster_block.hpp"

#include <armadillo>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace AI4BayesCode;

int G_PASS = 0, G_FAIL = 0;
void check(bool ok, const std::string& tag, const std::string& detail = "") {
    if (ok) { ++G_PASS; std::printf("  PASS  %s\n", tag.c_str()); }
    else    { ++G_FAIL; std::printf("  FAIL  %s  %s\n", tag.c_str(), detail.c_str()); }
}

bool same_bits(const arma::vec& a, const arma::vec& b) {
    return a.n_elem == b.n_elem &&
           std::memcmp(a.memptr(), b.memptr(), a.n_elem * sizeof(double)) == 0;
}

arma::vec fixed_vec(std::size_t n, double lo, double hi, std::uint64_t seed) {
    std::mt19937_64 g(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    arma::vec v(n);
    for (std::size_t e = 0; e < n; ++e) {
        const double r = u(g);
        v[e] = (r < 0.1) ? lo : (r < 0.2 ? hi : lo + (hi - lo) * u(g));
    }
    return v;
}

arma::vec field_vec(std::size_t n, std::size_t Q, std::uint64_t seed) {
    std::mt19937_64 g(seed);
    std::normal_distribution<double> z(0.0, 1.0);
    arma::vec f(n * Q);
    for (std::size_t k = 0; k < f.n_elem; ++k) {
        const double d = z(g);
        f[k] = -0.5 * d * d;
    }
    return f;
}

ising_cluster_block_config base_cfg(int L, std::size_t Q) {
    ising_cluster_block_config cfg;
    cfg.name = "x";
    cfg.n_vertices = static_cast<std::size_t>(L) * L;
    cfg.n_states = Q;
    cfg.edges = make_2d_lattice_edges(L, L, true, false);
    return cfg;
}

const double kBetas[] = {0.44, 0.44, 0.9, 0.0, 0.44, 1.3, -0.0, 0.9, 0.9, 2.5};

// T1: the three ways of supplying the same beta and delta give one chain.
void T1_paths_agree() {
    std::printf("T1  scalar / per-edge config / per-edge context paths\n");
    for (std::size_t Q : {2u, 3u}) {
        for (double delta : {1.0, 0.6, 0.0}) {
            for (bool with_field : {false, true}) {
                const int L = 8;
                ising_cluster_block_config a = base_cfg(L, Q);
                const std::size_t E = a.edges.n_cols, N = a.n_vertices;
                a.beta_key = "beta";
                a.delta_default = delta;
                if (with_field) a.field_key = "f";
                ising_cluster_block_config b = a;          // per-edge delta from config
                b.delta_edge = arma::vec(E, arma::fill::value(delta));
                ising_cluster_block_config c = a;          // per-edge beta, delta from context
                c.beta_edge_key = "be";
                c.delta_key = "de";
                ising_cluster_block A(a), B(b), C(c);
                std::mt19937_64 ra(7), rb(7), rc(7);
                bool ok = true;
                int t = 0;
                for (; t < 1500 && ok; ++t) {
                    const double beta = kBetas[(t / 7) % 10];
                    block_context ctx;
                    ctx["beta"] = arma::vec{beta};
                    if (with_field) ctx["f"] = field_vec(N, Q, 100 + t / 50);
                    A.set_context(ctx);
                    B.set_context(ctx);
                    ctx["be"] = arma::vec(E, arma::fill::value(beta));
                    ctx["de"] = arma::vec(E, arma::fill::value(delta));
                    C.set_context(ctx);
                    A.step(ra); B.step(rb); C.step(rc);
                    ok = same_bits(A.current(), B.current()) &&
                         same_bits(A.current(), C.current()) && ra == rb && ra == rc;
                }
                char tag[96];
                std::snprintf(tag, sizeof tag, "Q=%zu delta=%.1f field=%d", Q, delta,
                              with_field ? 1 : 0);
                check(ok, tag, ok ? "" : "chains differ at sweep " + std::to_string(t));
            }
        }
    }
}

// One sweep of a block constructed afresh from (cfg, state); returns state.
arma::vec fresh_sweep(ising_cluster_block_config cfg, const arma::vec& state,
                      const block_context& ctx, std::mt19937_64& rng) {
    cfg.initial_state = state;
    ising_cluster_block F(std::move(cfg));
    F.set_context(ctx);
    F.step(rng);
    return F.current();
}

// T2: a long-lived block vs a fresh block before every sweep.
void T2_long_lived_vs_fresh() {
    std::printf("T2  long-lived block vs fresh block every sweep\n");
    for (int variant = 0; variant < 3; ++variant) {
        const int L = 10;
        const std::size_t Q = 3;
        ising_cluster_block_config cfg = base_cfg(L, Q);
        const std::size_t E = cfg.edges.n_cols, N = cfg.n_vertices;
        cfg.beta_key = "beta";
        cfg.beta_edge_key = "be";
        cfg.delta_key = "de";
        cfg.delta_default = 0.7;
        if (variant != 1) cfg.delta_edge = fixed_vec(E, 0.0, 1.0, 11);
        if (variant != 0) cfg.beta_edge = fixed_vec(E, 0.0, 1.2, 12);
        arma::vec be = fixed_vec(E, 0.0, 1.5, 13), de = fixed_vec(E, 0.0, 1.0, 14);
        auto P = std::make_unique<ising_cluster_block>(cfg);
        std::mt19937_64 rng(21), ctl(22);
        std::uniform_int_distribution<int> pick(0, 9);
        bool ok = true;
        int t = 0;
        for (; t < 2000 && ok; ++t) {
            const int k = pick(ctl);
            block_context ctx;
            ctx["beta"] = arma::vec{kBetas[k]};
            if (k % 3 == 1) ctx["be"] = be;      // k = 1, 4, 7
            if (k % 2 == 0) ctx["de"] = de;      // k = 0, 2, 4, 6, 8
            if (k == 5) be[(7 * t) % E] = 0.1 * (t % 9);
            if (k == 8) de[(11 * t) % E] = 0.1 * (t % 11);
            std::mt19937_64 rng_f = rng;
            const arma::vec expect = fresh_sweep(cfg, P->current(), ctx, rng_f);
            P->set_context(ctx);
            P->step(rng);
            ok = same_bits(P->current(), expect) && rng == rng_f;
            if (t == 500) P = std::make_unique<ising_cluster_block>(*P);
            if (t == 1000) P = std::make_unique<ising_cluster_block>(std::move(*P));
            if (t == 1500) {
                ising_cluster_block other(cfg);
                other = *P;
                P = std::make_unique<ising_cluster_block>(std::move(other));
            }
            if (t == 1700) {
                std::mt19937_64 g(23);
                std::uniform_int_distribution<int> s(0, static_cast<int>(Q) - 1);
                arma::vec x(N);
                for (std::size_t i = 0; i < N; ++i) x[i] = s(g);
                P->set_current(x);
            }
        }
        const char* names[] = {"config delta_edge, scalar beta",
                               "config beta_edge, scalar delta",
                               "config beta_edge and delta_edge"};
        check(ok, names[variant],
              ok ? "" : "chains differ at sweep " + std::to_string(t - 1));
    }
}

// T3: field from a changing context key vs the same field in a fresh config.
void T3_field_in_place() {
    std::printf("T3  field from context key vs field in config\n");
    for (double delta : {1.0, 0.5}) {
        const int L = 8;
        const std::size_t Q = 3;
        ising_cluster_block_config cfg = base_cfg(L, Q);
        const std::size_t N = cfg.n_vertices;
        cfg.beta_default = 0.6;
        cfg.delta_default = delta;
        ising_cluster_block_config cfg_key = cfg;
        cfg_key.field_key = "f";
        ising_cluster_block P(cfg_key);
        std::mt19937_64 rng(31);
        bool ok = true;
        int t = 0;
        for (; t < 1500 && ok; ++t) {
            const arma::vec f = field_vec(N, Q, 200 + t / 40);
            block_context ctx;
            ctx["f"] = f;
            ising_cluster_block_config cfg_mat = cfg;
            cfg_mat.field = arma::reshape(f, N, Q);
            std::mt19937_64 rng_f = rng;
            const arma::vec expect = fresh_sweep(cfg_mat, P.current(), block_context(), rng_f);
            P.set_context(ctx);
            P.step(rng);
            ok = same_bits(P.current(), expect) && rng == rng_f;
        }
        char tag[64];
        std::snprintf(tag, sizeof tag, "delta=%.1f", delta);
        check(ok, tag, ok ? "" : "chains differ at sweep " + std::to_string(t - 1));
    }
}

// T4: no edges -> no residual coupling, so delta does not enter the sweep.
void T4_empty_edges() {
    std::printf("T4  empty edge list\n");
    for (bool with_field : {false, true}) {
        ising_cluster_block_config a;
        a.name = "x"; a.n_vertices = 12; a.n_states = 3;
        a.edges = arma::umat(2, 0);
        a.beta_default = 0.8;
        if (with_field) a.field = arma::reshape(field_vec(12, 3, 41), 12, 3);
        ising_cluster_block_config b = a;
        a.delta_default = 0.5;
        b.delta_default = 1.0;
        ising_cluster_block A(a), B(b);
        std::mt19937_64 ra(5), rb(5);
        bool ok = true;
        for (int t = 0; t < 500 && ok; ++t) {
            A.set_context(block_context()); B.set_context(block_context());
            A.step(ra); B.step(rb);
            ok = same_bits(A.current(), B.current()) && ra == rb;
        }
        check(ok, with_field ? "with field" : "no field");
    }
}

// T5: field (i, k) is entry i + k * n of the column-major field.
void T5_field_orientation() {
    std::printf("T5  field orientation\n");
    const int L = 6;
    for (std::size_t Q : {2u, 3u}) {
        for (int c = 0; c < 4; ++c) {
            const bool from_key = (c & 1) != 0;
            const bool case_c = (c & 2) != 0;   // beta > 0, delta = 0: no bonds, residual beta
            ising_cluster_block_config cfg = base_cfg(L, Q);
            const std::size_t N = cfg.n_vertices;
            arma::mat alpha(N, Q);
            for (std::size_t i = 0; i < N; ++i)
                for (std::size_t k = 0; k < Q; ++k)
                    alpha(i, k) = (k == i % Q) ? 0.0 : -1000.0;
            cfg.beta_default = case_c ? 0.5 : 0.0;
            cfg.delta_default = case_c ? 0.0 : 1.0;
            cfg.initial_state = arma::vec(N);
            for (std::size_t i = 0; i < N; ++i)
                cfg.initial_state[i] = static_cast<double>((i + 1) % Q);
            block_context ctx;
            if (from_key) { cfg.field_key = "f"; ctx["f"] = arma::vectorise(alpha); }
            else          { cfg.field = alpha; }
            ising_cluster_block B(cfg);
            std::mt19937_64 rng(51);
            B.set_context(ctx);
            B.step(rng);
            bool ok = true;
            for (std::size_t i = 0; i < N; ++i)
                ok = ok && B.current()[i] == static_cast<double>(i % Q);
            char tag[96];
            std::snprintf(tag, sizeof tag, "Q=%zu %s %s", Q, case_c ? "case C" : "case B",
                          from_key ? "context field" : "config field");
            check(ok, tag);
        }
    }
}

} // namespace

int main() {
    std::printf("====== ising_cluster_block bond probability / field paths ======\n");
    T1_paths_agree();
    T2_long_lived_vs_fresh();
    T3_field_in_place();
    T4_empty_edges();
    T5_field_orientation();
    std::printf("\n====== Summary: %d PASS / %d FAIL ======\n", G_PASS, G_FAIL);
    return (G_FAIL == 0) ? 0 : 1;
}

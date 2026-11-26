// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_examples_ICARSpatialGMRF.cpp
//
//  examples/ICARSpatialGMRF.cpp: set_current. Intercept, tau and sigma are
//  the state of the joint block (child 1), phi the state of the GMRF block
//  (child 0); y is data.
//    C1  set_current with a valid Intercept and tau and a phi of wrong
//        length throws and leaves every shared value and every block's
//        current value unchanged (bit for bit);
//    C2  the same for a valid phi and a y of wrong length;
//    C3  set_current of Intercept, tau, sigma and phi is reported by
//        get_current(), and the blocks hold the loaded values (the joint
//        block up to rounding, rel. 1e-12: it holds tau and sigma through
//        their logarithms).
//  Posterior predictive y_rep_t ~ N(Intercept + phi[node_idx[t]], sigma^2):
//    C4  keep_history = true: predict_at({}) returns only y_rep, one row per
//        stored draw; row d equals, bit for bit, Intercept_d +
//        phi_d[node_idx] + sigma_d * e with Intercept_d, sigma_d, phi_d row d
//        of get_history() and e the N(0, 1) values of a copy of predict_rng_
//        (a new distribution object per draw, as the refresher draws them);
//        with both blocks frozen and Intercept loaded as -5, then 5, the rows
//        of the held draws are centred on -5 and 5 respectively;
//    C5  last_draw_only = true, and a keep_history = false model, return one
//        row drawn from the current state (bit for bit as in C4); a
//        non-empty new_data throws std::runtime_error;
//    C6  clear_history() (clear_history_contract.hpp): after k1 sweeps,
//        clear_history() and k2 sweeps, get_history() equals the last k2
//        rows of a run without the call and y_rep has k2 rows;
//    C7  posterior predictive check on the data of the @example blocks (5x5
//        lattice, 5 observations per node): the predictive p-values of
//        mean(y) and var(y) over the stored draws lie in [0.025, 0.975], and
//        the mean and variance of all y_rep values are within 0.1 and 10% of
//        those of y.
// ============================================================================
#include <armadillo>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "AI4BayesCode/composite_block.hpp"

#define main icar_spatial_gmrf_demo_main
#include "../examples/ICARSpatialGMRF.cpp"
#undef main

#include "clear_history_contract.hpp"

// An explicit instantiation may name a private member: this gives the test
// access to the wrapper's composite_block.
template <typename Tag, typename Tag::type Member>
struct expose_member {
    friend typename Tag::type member_of(Tag) { return Member; }
};
struct impl_tag {
    using type = std::unique_ptr<AI4BayesCode::composite_block> ICARSpatialGMRF::*;
    friend type member_of(impl_tag);
};
template struct expose_member<impl_tag, &ICARSpatialGMRF::impl_>;
struct prng_tag {
    using type = std::mt19937_64 ICARSpatialGMRF::*;
    friend type member_of(prng_tag);
};
template struct expose_member<prng_tag, &ICARSpatialGMRF::predict_rng_>;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

AI4BayesCode::composite_block& impl(const ICARSpatialGMRF& m) {
    return *(m.*member_of(impl_tag{}));
}

bool same_bits(const arma::vec& a, const arma::vec& b) {
    return a.n_elem == b.n_elem &&
           std::memcmp(a.memptr(), b.memptr(), a.n_elem * sizeof(double)) == 0;
}

// Every shared value and every block's current value.
struct state_t {
    std::map<std::string, arma::vec> shared;
    std::vector<arma::vec> blocks;
};
state_t state_of(const ICARSpatialGMRF& m) {
    state_t s;
    for (const auto& kv : impl(m).data().values()) s.shared.emplace(kv.first, kv.second);
    for (std::size_t i = 0; i < impl(m).n_children(); ++i)
        s.blocks.push_back(impl(m).child(i).current());
    return s;
}
bool same_state(const state_t& a, const state_t& b) {
    if (a.shared.size() != b.shared.size() || a.blocks.size() != b.blocks.size())
        return false;
    for (const auto& kv : a.shared) {
        auto it = b.shared.find(kv.first);
        if (it == b.shared.end() || !same_bits(kv.second, it->second)) return false;
    }
    for (std::size_t i = 0; i < a.blocks.size(); ++i)
        if (!same_bits(a.blocks[i], b.blocks[i])) return false;
    return true;
}
// set_current(s) throws, and the state is unchanged.
bool throws_unchanged(ICARSpatialGMRF& m, const AI4BayesCode::state_map& s) {
    const state_t before = state_of(m);
    bool threw = false;
    try { m.set_current(s); } catch (const std::exception&) { threw = true; }
    return threw && same_state(before, state_of(m));
}

bool same_mat(const arma::mat& a, const arma::mat& b) {
    return a.n_rows == b.n_rows && a.n_cols == b.n_cols &&
           std::memcmp(a.memptr(), b.memptr(), a.n_elem * sizeof(double)) == 0;
}

// y_rep for the draws in the rows of I (Intercept), S (sigma) and Phi (phi),
// drawn as the "y_rep" refresher draws it: for each row a new N(0, 1)
// distribution on rng, then y_rep(d, t) = I(d) + Phi(d, node0[t]) + S(d) * e.
arma::mat yrep_ref(const arma::mat& I, const arma::mat& S, const arma::mat& Phi,
                   const arma::vec& node0, std::mt19937_64 rng) {
    arma::mat Y(I.n_rows, node0.n_elem);
    for (arma::uword d = 0; d < I.n_rows; ++d) {
        std::normal_distribution<double> nd(0.0, 1.0);
        for (arma::uword t = 0; t < node0.n_elem; ++t)
            Y(d, t) = I(d, 0) + Phi(d, static_cast<arma::uword>(node0[t]))
                      + S(d, 0) * nd(rng);
    }
    return Y;
}

const std::mt19937_64& predict_rng(const ICARSpatialGMRF& m) {
    return m.*member_of(prng_tag{});
}

// C6: a keep_history model whose per-draw buffers are get_history() and the
// y_rep rows of predict_at({}).
clear_history_check::model contract_model(const arma::vec& y, const arma::vec& node,
                                          int n_nodes, const arma::vec& ei,
                                          const arma::vec& ej) {
    auto w = std::make_shared<ICARSpatialGMRF>(y, node, n_nodes, ei, ej, 23, true);
    ICARSpatialGMRF* p = w.get();
    clear_history_check::model c;
    c.step    = [p](int n) { p->step(n); };
    c.clear   = [p] { p->clear_history(); };
    c.history = [p](clear_history_check::buffers& b) {
        for (const auto& kv : p->get_history()) b.mats[kv.first] = kv.second;
    };
    c.extra   = [p](clear_history_check::buffers& b) {
        for (const auto& kv : p->predict_at(AI4BayesCode::state_map{}))
            b.rows_only["predict_at " + kv.first] = kv.second;
    };
    c.owners.push_back(w);
    return c;
}

}  // namespace

int main() {
    std::printf("=== test_fix_examples_ICARSpatialGMRF ===\n");
    std::vector<int> ei_v, ej_v;
    int n_nodes = 0;
    build_lattice(3, ei_v, ej_v, n_nodes);
    const std::size_t N = static_cast<std::size_t>(n_nodes);
    arma::vec edges_i(ei_v.size()), edges_j(ej_v.size());
    for (std::size_t e = 0; e < ei_v.size(); ++e) {
        edges_i[e] = static_cast<double>(ei_v[e]);
        edges_j[e] = static_cast<double>(ej_v[e]);
    }
    std::mt19937_64 gen(31u);
    std::normal_distribution<double> nd(0.0, 1.0);
    arma::vec node(N), y(N);
    for (std::size_t t = 0; t < N; ++t) {
        node[t] = static_cast<double>(t) + 1.0;
        y[t] = 4.0 + 0.3 * static_cast<double>(t % 3) + 0.5 * nd(gen);
    }
    ICARSpatialGMRF m(y, node, n_nodes, edges_i, edges_j, 7, false);
    m.step(3);

    // C1
    AI4BayesCode::state_map s1;
    s1["Intercept"] = arma::vec{2.0};
    s1["tau"] = arma::vec{3.0};
    s1["phi"] = arma::vec(N + 1, arma::fill::zeros);
    check(throws_unchanged(m, s1),
          "C1 a valid Intercept and tau with a phi of wrong length throw and change nothing");

    // C2
    AI4BayesCode::state_map s2;
    s2["phi"] = arma::linspace<arma::vec>(-1.0, 1.0, N);
    s2["y"] = arma::vec(N + 2, arma::fill::ones);
    check(throws_unchanged(m, s2),
          "C2 a valid phi with a y of wrong length throws and changes nothing");

    // C3
    AI4BayesCode::state_map s3;
    s3["Intercept"] = arma::vec{2.0};
    s3["tau"] = arma::vec{3.0};
    s3["sigma"] = arma::vec{0.75};
    s3["phi"] = arma::linspace<arma::vec>(-1.0, 1.0, N);
    m.set_current(s3);
    const auto g3 = m.get_current();
    bool c3 = same_bits(impl(m).child(0).current(), s3.at("phi"));
    for (const auto& kv : s3) c3 = c3 && same_bits(g3.at(kv.first), kv.second);
    // The joint block holds tau and sigma through their logarithms, so its
    // natural values equal the loaded ones up to rounding (rel. 1e-12).
    const arma::vec& jb = impl(m).child(1).current();
    const arma::vec want{2.0, 3.0, 0.75};
    c3 = c3 && jb.n_elem == 3 &&
         arma::max(arma::abs(jb - want) / arma::abs(want)) <= 1e-12;
    check(c3, "C3 the loaded values are reported and held by the blocks");

    // C4
    const arma::vec node0 = node - 1.0;
    ICARSpatialGMRF h(y, node, n_nodes, edges_i, edges_j, 19, true);
    {
        h.step(12);
        impl(h).freeze(std::vector<std::string>{"phi", "params_joint"}, true);
        AI4BayesCode::state_map lo, hi;
        lo["Intercept"] = arma::vec{-5.0};
        lo["sigma"] = arma::vec{0.2};
        lo["phi"] = arma::vec(N, arma::fill::zeros);
        hi = lo;
        hi["Intercept"] = arma::vec{5.0};
        h.set_current(lo);
        h.step(4);
        h.set_current(hi);
        h.step(4);
        const auto H = h.get_history();
        const std::mt19937_64 rng0 = predict_rng(h);
        const auto p = h.predict_at(AI4BayesCode::state_map{});
        const bool shape_ok = p.size() == 1 && p.count("y_rep") &&
                              p.at("y_rep").n_rows == 20 && p.at("y_rep").n_cols == N &&
                              H.at("Intercept").n_rows == 20;
        std::printf("C4 history mode: y_rep %llux%llu, %llu stored draws\n",
                    static_cast<unsigned long long>(shape_ok ? p.at("y_rep").n_rows : 0),
                    static_cast<unsigned long long>(shape_ok ? p.at("y_rep").n_cols : 0),
                    static_cast<unsigned long long>(H.at("Intercept").n_rows));
        check(shape_ok, "C4 predict_at({}) returns y_rep only, one row per stored draw");
        check(shape_ok && same_mat(p.at("y_rep"),
                                   yrep_ref(H.at("Intercept"), H.at("sigma"), H.at("phi"),
                                            node0, rng0)),
              "C4 row d = Intercept_d + phi_d[node_idx] + sigma_d * e from stored draw d "
              "(bit for bit)");
        bool held_ok = shape_ok;
        const double tol = 5.0 * 0.2 / std::sqrt(static_cast<double>(N));
        for (arma::uword d = 12; held_ok && d < 20; ++d) {
            const double centre = (d < 16) ? -5.0 : 5.0;
            held_ok = std::abs(arma::mean(p.at("y_rep").row(d)) - centre) < tol;
        }
        check(held_ok, "C4 held draws with Intercept -5, then 5: rows centred on -5, then 5");
    }

    // C5
    {
        const std::mt19937_64 rng1 = predict_rng(h);
        const auto p1 = h.predict_at(AI4BayesCode::state_map{}, true);
        const auto g = h.get_current();
        check(p1.size() == 1 && p1.count("y_rep") &&
              same_mat(p1.at("y_rep"), yrep_ref(arma::mat(g.at("Intercept")),
                                                arma::mat(g.at("sigma")),
                                                arma::mat(g.at("phi").t()), node0, rng1)),
              "C5 last_draw_only = true: one row, drawn from the current state");
        const std::mt19937_64 rng2 = predict_rng(m);   // keep_history = false
        const auto p2 = m.predict_at(AI4BayesCode::state_map{});
        const auto gm = m.get_current();
        check(p2.size() == 1 && p2.count("y_rep") &&
              same_mat(p2.at("y_rep"), yrep_ref(arma::mat(gm.at("Intercept")),
                                                arma::mat(gm.at("sigma")),
                                                arma::mat(gm.at("phi").t()), node0, rng2)),
              "C5 keep_history = false: one row, drawn from the current state");
        AI4BayesCode::state_map nd;
        nd["node_idx"] = node;
        bool runtime_error = false;
        try { (void)m.predict_at(nd); }
        catch (const std::runtime_error&) { runtime_error = true; }
        catch (const std::exception&) {}
        check(runtime_error, "C5 a non-empty new_data throws std::runtime_error");
    }

    // C6
    clear_history_check::run("C6 clear_history() and predict_at",
                             [&] { return contract_model(y, node, n_nodes, edges_i, edges_j); },
                             40, 30);

    // C7: data generated as in the @example blocks (5x5 lattice, centred
    // field 2 sin(pi x) sin(pi v) + (x - 0.5), 5 observations per node,
    // Intercept 4, sigma 0.5), with normal draws from std::mt19937_64.
    {
        std::vector<int> ei7, ej7;
        int n7 = 0;
        build_lattice(5, ei7, ej7, n7);
        arma::vec e_i(ei7.size()), e_j(ej7.size());
        for (std::size_t e = 0; e < ei7.size(); ++e) {
            e_i[e] = static_cast<double>(ei7[e]);
            e_j[e] = static_cast<double>(ej7[e]);
        }
        arma::vec phi7(static_cast<std::size_t>(n7));
        for (std::size_t k = 0; k < phi7.n_elem; ++k) {
            const double x = static_cast<double>(k % 5) / 4.0;
            const double v = static_cast<double>(k / 5) / 4.0;
            phi7[k] = 2.0 * std::sin(3.14159265 * x) * std::sin(3.14159265 * v)
                      + (x - 0.5);
        }
        phi7 -= arma::mean(phi7);
        const std::size_t T7 = 5 * phi7.n_elem;
        arma::vec node7(T7), y7(T7);
        std::mt19937_64 g7(20260621u);
        std::normal_distribution<double> z7(0.0, 0.5);
        for (std::size_t t = 0; t < T7; ++t) {
            node7[t] = static_cast<double>(t / 5) + 1.0;
            y7[t] = 4.0 + phi7[t / 5] + z7(g7);
        }
        ICARSpatialGMRF q(y7, node7, n7, e_i, e_j, 7, true);
        q.step(1000);
        q.clear_history();
        q.step(2000);
        const arma::mat Y = q.predict_at(AI4BayesCode::state_map{}).at("y_rep");
        const double my = arma::mean(y7), vy = arma::var(y7);
        double p_mean = 0.0, p_var = 0.0;
        for (arma::uword d = 0; d < Y.n_rows; ++d) {
            p_mean += (arma::mean(Y.row(d)) >= my) ? 1.0 : 0.0;
            p_var  += (arma::var(Y.row(d)) >= vy) ? 1.0 : 0.0;
        }
        p_mean /= static_cast<double>(Y.n_rows);
        p_var  /= static_cast<double>(Y.n_rows);
        const double mY = arma::mean(arma::vectorise(Y));
        const double vY = arma::var(arma::vectorise(Y));
        std::printf("C7 y: mean %.4f var %.4f | y_rep (%llu draws): mean %.4f var %.4f | "
                    "p-values: mean %.3f var %.3f\n", my, vy,
                    static_cast<unsigned long long>(Y.n_rows), mY, vY, p_mean, p_var);
        check(Y.n_rows == 2000 && Y.n_cols == T7, "C7 y_rep has one row per stored draw");
        check(p_mean >= 0.025 && p_mean <= 0.975 && p_var >= 0.025 && p_var <= 0.975,
              "C7 predictive p-values of mean(y) and var(y) in [0.025, 0.975]");
        check(std::abs(mY - my) < 0.1 && std::abs(vY / vy - 1.0) < 0.1,
              "C7 mean and variance of y_rep within 0.1 and 10% of those of y");
    }

    const int n_fail = g_fail + clear_history_check::failures();
    std::printf("%s (%d failure%s)\n", n_fail ? "FAILED" : "ALL PASSED", n_fail,
                n_fail == 1 ? "" : "s");
    return n_fail ? 1 : 0;
}

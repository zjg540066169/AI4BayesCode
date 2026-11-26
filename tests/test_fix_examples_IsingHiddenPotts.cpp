// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_examples_IsingHiddenPotts.cpp
//
//  examples/IsingHiddenPotts.cpp: z (labels) is the state of the Ising
//  cluster block, which rounds a loaded z; get_current() reports the shared
//  "z".
//    C1  after set_current(z = 0.6 everywhere), the shared z equals the
//        block's labels (all 1);
//    C2  set_current with a z whose last entry is -0.6 (rounds to -1)
//        throws and leaves the block's labels and the shared z unchanged;
//    C3  an entry of -1, which passes the example's own check (z + 0.5
//        truncated toward zero is 0 or 1), throws std::invalid_argument, the
//        exception of ising_cluster_block::set_current; an entry of 2 throws
//        std::runtime_error from the example's own check.
//  Posterior predictive y_rep_i ~ N(mu_{z_i}, sigma^2):
//    C4  keep_history = true: predict_at({}) returns only y_rep, one row per
//        stored draw; row d equals, bit for bit, mu_{z_d} + sigma * e with z_d
//        row d of get_history()$z and e the N(0, 1) values of a copy of
//        predict_rng_ (a new distribution object per draw, as the refresher
//        draws them); with z frozen and loaded all-0, then all-1, the rows of
//        the held draws are centred on mu0 and mu1 respectively;
//    C5  last_draw_only = true, and a keep_history = false model, return one
//        row drawn from the current labels (bit for bit as in C4); a
//        non-empty new_data throws std::runtime_error;
//    C6  clear_history() (clear_history_contract.hpp): after k1 sweeps,
//        clear_history() and k2 sweeps, get_history() equals the last k2
//        rows of a run without the call and y_rep has k2 rows;
//    C7  posterior predictive check on the data of the standalone demo:
//        the predictive p-values of mean(y) and var(y) over the stored draws
//        lie in [0.025, 0.975], and the mean and variance of all y_rep
//        values are within 0.1 and 10% of those of y.
// ============================================================================
#include <armadillo>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "AI4BayesCode/composite_block.hpp"

#define main ising_hidden_potts_demo_main
#include "../examples/IsingHiddenPotts.cpp"
#undef main

#include "clear_history_contract.hpp"

// An explicit instantiation may name a private member: this gives the test
// read access to the wrapper's composite_block.
template <typename Tag, typename Tag::type Member>
struct expose_member {
    friend typename Tag::type member_of(Tag) { return Member; }
};
struct impl_tag {
    using type = std::unique_ptr<AI4BayesCode::composite_block> IsingHiddenPotts::*;
    friend type member_of(impl_tag);
};
template struct expose_member<impl_tag, &IsingHiddenPotts::impl_>;
struct prng_tag {
    using type = std::mt19937_64 IsingHiddenPotts::*;
    friend type member_of(prng_tag);
};
template struct expose_member<prng_tag, &IsingHiddenPotts::predict_rng_>;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

bool same_bits(const arma::vec& a, const arma::vec& b) {
    return a.n_elem == b.n_elem &&
           std::memcmp(a.memptr(), b.memptr(), a.n_elem * sizeof(double)) == 0;
}

bool same_mat(const arma::mat& a, const arma::mat& b) {
    return a.n_rows == b.n_rows && a.n_cols == b.n_cols &&
           std::memcmp(a.memptr(), b.memptr(), a.n_elem * sizeof(double)) == 0;
}

// y_rep for the labels in the rows of Z, drawn as the "y_rep" refresher draws
// it: for each row a new N(0, 1) distribution on rng, then
// y_rep(d, i) = mu_{Z(d, i)} + sigma * e for i = 0..N-1.
arma::mat yrep_ref(const arma::mat& Z, double mu0, double mu1, double sigma,
                   std::mt19937_64 rng) {
    arma::mat Y(Z.n_rows, Z.n_cols);
    for (arma::uword d = 0; d < Z.n_rows; ++d) {
        std::normal_distribution<double> nd(0.0, 1.0);
        for (arma::uword i = 0; i < Z.n_cols; ++i)
            Y(d, i) = ((Z(d, i) > 0.5) ? mu1 : mu0) + sigma * nd(rng);
    }
    return Y;
}

const std::mt19937_64& predict_rng(const IsingHiddenPotts& m) {
    return m.*member_of(prng_tag{});
}

// C6: a keep_history model whose per-draw buffers are get_history() and the
// y_rep rows of predict_at({}).
clear_history_check::model contract_model() {
    auto w = std::make_shared<IsingHiddenPotts>(8, 0.6, 0.5, -1.0, 1.0, 0.8,
                                                2026, 17, 0, true);
    IsingHiddenPotts* p = w.get();
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
    std::printf("=== test_fix_examples_IsingHiddenPotts ===\n");
    IsingHiddenPotts m(8, 0.6, 0.5, -1.0, 1.0, 0.8, 2026, 7, 0, false);
    const std::size_t N = m.n_sites();
    m.step(3);
    auto& c = *(m.*member_of(impl_tag{}));

    AI4BayesCode::state_map s;
    s["z"] = arma::vec(N, arma::fill::value(0.6));
    m.set_current(s);
    std::printf("C1 set_current(z = 0.6): block z[0] %.1f, reported z[0] %.1f\n",
                c.child(0).current()[0], m.get_current().at("z")[0]);
    check(same_bits(c.child(0).current(), m.get_current().at("z")) &&
          c.child(0).current()[0] == 1.0,
          "C1 the shared z equals the block's labels");

    m.step(2);
    const arma::vec zb = c.child(0).current(), zs = m.get_current().at("z");
    arma::vec bad(N, arma::fill::ones);
    bad[N - 1] = -0.6;
    AI4BayesCode::state_map b;
    b["z"] = bad;
    bool threw = false;
    try { m.set_current(b); } catch (const std::exception&) { threw = true; }
    check(threw && same_bits(zb, c.child(0).current()) && same_bits(zs, m.get_current().at("z")),
          "C2 a z entry rounding to -1 is rejected and nothing is written");

    {
        arma::vec z_m1(N, arma::fill::ones), z_2(N, arma::fill::ones);
        z_m1[0] = -1.0;
        z_2[0] = 2.0;
        AI4BayesCode::state_map sm1, s2;
        sm1["z"] = z_m1;
        s2["z"] = z_2;
        bool invalid_argument = false, runtime_error = false;
        try { m.set_current(sm1); }
        catch (const std::invalid_argument&) { invalid_argument = true; }
        catch (const std::exception&) {}
        try { m.set_current(s2); }
        catch (const std::invalid_argument&) {}
        catch (const std::runtime_error&) { runtime_error = true; }
        catch (const std::exception&) {}
        check(invalid_argument && runtime_error,
              "C3 z = -1 throws std::invalid_argument and z = 2 std::runtime_error");
    }

    // C4
    const double mu0 = -1.0, mu1 = 1.0, sd = 0.8;
    IsingHiddenPotts h(8, 0.6, 0.5, mu0, mu1, sd, 2026, 13, 0, true);
    {
        auto& hc = *(h.*member_of(impl_tag{}));
        h.step(10);
        hc.freeze(std::vector<std::string>{"z"}, true);
        AI4BayesCode::state_map s0, s1;
        s0["z"] = arma::vec(N, arma::fill::zeros);
        s1["z"] = arma::vec(N, arma::fill::ones);
        h.set_current(s0);
        h.step(6);
        h.set_current(s1);
        h.step(6);
        const arma::mat Z = h.get_history().at("z");
        const std::mt19937_64 rng0 = predict_rng(h);
        const auto p = h.predict_at(AI4BayesCode::state_map{});
        const bool shape_ok = p.size() == 1 && p.count("y_rep") &&
                              p.at("y_rep").n_rows == 22 && p.at("y_rep").n_cols == N &&
                              Z.n_rows == 22;
        std::printf("C4 history mode: y_rep %llux%llu, get_history()$z %llux%llu\n",
                    static_cast<unsigned long long>(shape_ok ? p.at("y_rep").n_rows : 0),
                    static_cast<unsigned long long>(shape_ok ? p.at("y_rep").n_cols : 0),
                    static_cast<unsigned long long>(Z.n_rows),
                    static_cast<unsigned long long>(Z.n_cols));
        check(shape_ok, "C4 predict_at({}) returns y_rep only, one row per stored draw");
        check(shape_ok && same_mat(p.at("y_rep"), yrep_ref(Z, mu0, mu1, sd, rng0)),
              "C4 row d = mu_{z_d} + sigma * e, z_d the labels of stored draw d (bit for bit)");
        bool held_ok = shape_ok;
        const double tol = 5.0 * sd / std::sqrt(static_cast<double>(N));
        for (arma::uword d = 10; held_ok && d < 22; ++d) {
            const double centre = (d < 16) ? mu0 : mu1;
            held_ok = std::abs(arma::mean(p.at("y_rep").row(d)) - centre) < tol;
        }
        check(held_ok, "C4 held draws with z all-0, then all-1: rows centred on mu0, then mu1");
    }

    // C5
    {
        const std::mt19937_64 rng1 = predict_rng(h);
        const auto p1 = h.predict_at(AI4BayesCode::state_map{}, true);
        const arma::mat zc = h.get_current().at("z").t();
        check(p1.size() == 1 && p1.count("y_rep") &&
              same_mat(p1.at("y_rep"), yrep_ref(zc, mu0, mu1, sd, rng1)),
              "C5 last_draw_only = true: one row, drawn from the current labels");
        const std::mt19937_64 rng2 = predict_rng(m);   // keep_history = false
        const auto p2 = m.predict_at(AI4BayesCode::state_map{});
        const arma::mat zm = m.get_current().at("z").t();
        check(p2.size() == 1 && p2.count("y_rep") &&
              same_mat(p2.at("y_rep"), yrep_ref(zm, -1.0, 1.0, 0.8, rng2)),
              "C5 keep_history = false: one row, drawn from the current labels");
        AI4BayesCode::state_map nd;
        nd["z"] = arma::vec(N, arma::fill::zeros);
        bool runtime_error = false;
        try { (void)m.predict_at(nd); }
        catch (const std::runtime_error&) { runtime_error = true; }
        catch (const std::exception&) {}
        check(runtime_error, "C5 a non-empty new_data throws std::runtime_error");
    }

    // C6
    clear_history_check::run("C6 clear_history() and predict_at", contract_model, 40, 30);

    // C7: the data of the standalone demo; y is regenerated as the
    // constructor generates it (same engine, seed and order of draws).
    {
        const double a0 = 0.0, a1 = 2.0, s = 1.0;
        IsingHiddenPotts q(16, 0.7, 0.6, a0, a1, s, 2024, 42, 0, true);
        q.step(500);
        q.clear_history();
        q.step(2000);
        const arma::mat Y = q.predict_at(AI4BayesCode::state_map{}).at("y_rep");
        const arma::vec& zt = q.true_z();
        std::mt19937_64 dgen(2024);
        std::normal_distribution<double> noise(0.0, s);
        arma::vec y(zt.n_elem);
        for (arma::uword i = 0; i < zt.n_elem; ++i)
            y[i] = ((zt[i] > 0.5) ? a1 : a0) + noise(dgen);
        const double my = arma::mean(y), vy = arma::var(y);
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
        check(Y.n_rows == 2000 && Y.n_cols == zt.n_elem, "C7 y_rep has one row per stored draw");
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

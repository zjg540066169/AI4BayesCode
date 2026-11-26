// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_examples_ODE_SIR.cpp
//
//  The derived key "I_traj" (infected compartment of the SIR solution at
//  t_obs) of examples/ODE_SIR.cpp, from which predict_at({}) draws
//  I_rep = I_traj * exp(sigma * N(0, 1)).
//    C1  after construction, "I_traj" equals the solution at the initial
//        theta, and log(I_rep / I_traj) / sigma from predict_at({}) is
//        centred on 0;
//    C2  after one sweep and set_current of beta, gamma and sigma, the same
//        holds at the loaded theta;
//    C3  set_current(get_current()) after 5 more sweeps returns the same
//        get_current() values, with "I_traj" exact;
//    C4  after one sweep, set_current(beta = 1e5) and set_current(beta =
//        +Inf), at which the ODE solver stops with an error: either the call
//        throws and leaves every shared value and the joint block's current
//        value unchanged (bit for bit), or "I_traj" equals the solution at the
//        loaded theta;
//    C5  set_current with an empty beta, gamma or sigma throws and leaves
//        get_current() unchanged.
//  predict_at with new t_obs / y0 (history mode unless stated):
//    C6  predict_at({}), predict_at({t_obs}) and predict_at({y0, t_obs}) with
//        the training values give the same I_traj and y_rep bit for bit;
//    C7  at new times starting after 0 and a new y0: one row per stored
//        draw; row d of I_traj equals sir_I_at(y0, t, theta_d, 1) bit for bit
//        and a solve with rtol = atol = 1e-10 to 1e-5 relative; y_rep[d, i]
//        equals I_traj[d, i] * exp(sigma_d * z) bit for bit, with z the
//        standard normals of a copy of predict_rng_ (a new
//        std::normal_distribution per draw, as in the y_rep refresher); the
//        call leaves get_history(), get_current() and the shared data
//        unchanged;
//    C8  at times 0..9 (a prefix of the training times): I_traj[d, 0] = I0;
//    C9  single-draw path (keep_history = false, and last_draw_only = true):
//        one row from the current theta and sigma, the same formulas;
//    C10 after clear_history() (no stored draw): one row from the current
//        state;
//    C11 invalid t_obs / y0 / keys throw std::invalid_argument and leave the
//        model unchanged;
//    C12 accuracy at new times (the solver step is capped at the largest
//        training spacing, 1 here): for every stored draw, with the count y0
//        (990, 10, 0) and the fraction y0 (0.99, 0.01, 0), I_traj at
//        t = 30, 60, 90, 120 requested alone agrees with the same times
//        within t = 1..120, and both agree with a classical RK4 solve with
//        step 1/64, to a relative 1e-4; the fraction y0 gives 1/1000 of the
//        count y0 to a relative 1e-4; at the training times history-mode
//        I_traj equals ode::rk45(sir_rhs, y0, t_obs, theta_d, 1e-6, 1e-6)
//        (the solve without the cap) bit for bit, for both y0.
//  The solution is recomputed in this test with the example's own sir_rhs
//  and solver settings (relative tolerance 1e-12). Centred: the mean of
//  log(I_rep / I_traj) / sigma over the 15 times lies within
//  5 / sqrt(15) of 0.
// ============================================================================
#include <armadillo>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "AI4BayesCode/composite_block.hpp"

#define main ode_sir_demo_main
#include "../examples/ODE_SIR.cpp"
#undef main

// An explicit instantiation may name a private member: this gives the test
// read access to the wrapper's composite_block.
template <typename Tag, typename Tag::type Member>
struct expose_member {
    friend typename Tag::type member_of(Tag) { return Member; }
};
struct impl_tag {
    using type = std::unique_ptr<AI4BayesCode::composite_block> ODE_SIR::*;
    friend type member_of(impl_tag);
};
template struct expose_member<impl_tag, &ODE_SIR::impl_>;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

const AI4BayesCode::shared_data_t& data_of(const ODE_SIR& m) {
    return (m.*member_of(impl_tag{}))->data();
}

arma::vec traj_from_inputs(const ODE_SIR& m) {
    const auto& d = data_of(m);
    const arma::mat y = AI4BayesCode::ode::rk45(sir_rhs, d.get("y0"), d.get("t_obs"),
                                                d.get("theta"), 1e-6, 1e-6);
    return y.col(1);
}

double traj_error(const ODE_SIR& m) {
    const arma::vec& a = data_of(m).get("I_traj");
    const arma::vec b = traj_from_inputs(m);
    if (a.n_elem != b.n_elem) return INFINITY;
    double e = 0.0;
    for (std::size_t i = 0; i < a.n_elem; ++i)
        e = std::max(e, std::abs(a[i] - b[i]) / std::max(1.0, std::abs(b[i])));
    return e;
}

double yrep_z(const ODE_SIR& m) {
    const arma::mat yr = m.predict_at(AI4BayesCode::state_map{}).at("y_rep");
    const arma::vec ref = traj_from_inputs(m);
    const double sigma = data_of(m).get("sigma")[0];
    double s = 0.0;
    for (std::size_t k = 0; k < ref.n_elem; ++k) s += std::log(yr[k] / ref[k]) / sigma;
    return s / std::sqrt(static_cast<double>(ref.n_elem));
}

bool same_bits(const arma::vec& a, const arma::vec& b) {
    return a.n_elem == b.n_elem &&
           std::memcmp(a.memptr(), b.memptr(), a.n_elem * sizeof(double)) == 0;
}

bool same_bits_mat(const arma::mat& a, const arma::mat& b) {
    return a.n_rows == b.n_rows && a.n_cols == b.n_cols &&
           std::memcmp(a.memptr(), b.memptr(), a.n_elem * sizeof(double)) == 0;
}

bool same_maps(const AI4BayesCode::history_map& a, const AI4BayesCode::history_map& b) {
    if (a.size() != b.size()) return false;
    for (const auto& kv : a) {
        auto it = b.find(kv.first);
        if (it == b.end() || !same_bits_mat(kv.second, it->second)) return false;
    }
    return true;
}

// The stream of predict_rng_ of a model built with rng_seed (nonzero).
std::mt19937_64 predict_stream(int rng_seed) {
    return std::mt19937_64{static_cast<std::uint64_t>(rng_seed) ^ 0x9E3779B97F4A7C15ULL};
}

// Checks rows of I_traj / y_rep against the per-draw formulas: row r uses
// theta = thetas.row(r), sigma = sigmas(r, 0), and one new normal_distribution
// per row on `g`. Returns {max bitwise mismatches, max rel error vs the
// rtol = atol = 1e-10 solve}.
struct row_check { int I_mismatch = 0; int y_mismatch = 0; double rel_err = 0.0; };
row_check check_rows(const arma::mat& I_traj, const arma::mat& y_rep,
                     const arma::mat& thetas, const arma::mat& sigmas,
                     const arma::vec& y0, const arma::vec& t, std::mt19937_64& g) {
    row_check rc;
    arma::vec ts = t;
    if (t[0] > 0.0) { ts.set_size(t.n_elem + 1); ts[0] = 0.0; ts.subvec(1, t.n_elem) = t; }
    const arma::uword off = ts.n_elem - t.n_elem;
    for (arma::uword r = 0; r < I_traj.n_rows; ++r) {
        const arma::vec th = thetas.row(r).t();
        const arma::vec ref = sir_I_at(y0, t, th, 1.0);   // training spacing 1
        const arma::vec I_r = I_traj.row(r).t();
        if (!same_bits(I_r, ref)) ++rc.I_mismatch;
        const arma::mat tight = AI4BayesCode::ode::rk45(sir_rhs, y0, ts, th, 1e-10, 1e-10);
        for (arma::uword i = 0; i < t.n_elem; ++i)
            rc.rel_err = std::max(rc.rel_err, std::abs(I_r[i] - tight(i + off, 1)) /
                                                  std::max(1e-300, std::abs(tight(i + off, 1))));
        std::normal_distribution<double> nd(0.0, 1.0);
        for (arma::uword i = 0; i < t.n_elem; ++i) {
            const double expect = I_r[i] * std::exp(sigmas(r, 0) * nd(g));
            if (std::memcmp(&expect, &y_rep(r, i), sizeof(double)) != 0) ++rc.y_mismatch;
        }
    }
    return rc;
}

// I(t_i) of the SIR solution from y0 at time 0 by classical RK4 with step
// h = 1/64 (each t_i a multiple of h): the reference of C12.
arma::vec rk4_I(const arma::vec& y0, const arma::vec& t, const arma::vec& th) {
    const double h = 1.0 / 64.0;
    arma::vec out(t.n_elem), y = y0;
    long long k = 0;
    for (arma::uword i = 0; i < t.n_elem; ++i) {
        const long long k_end = std::llround(t[i] / h);
        for (; k < k_end; ++k) {
            const double tk = static_cast<double>(k) * h;
            const arma::vec k1 = sir_rhs(tk, y, th);
            const arma::vec k2 = sir_rhs(tk + 0.5 * h, y + 0.5 * h * k1, th);
            const arma::vec k3 = sir_rhs(tk + 0.5 * h, y + 0.5 * h * k2, th);
            const arma::vec k4 = sir_rhs(tk + h, y + h * k3, th);
            y += (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        }
        out[i] = y[1];
    }
    return out;
}

double max_rel(const arma::vec& a, const arma::vec& b) {
    double e = 0.0;
    for (arma::uword i = 0; i < a.n_elem; ++i)
        e = std::max(e, std::abs(a[i] - b[i]) / std::abs(b[i]));
    return a.n_elem == b.n_elem ? e : INFINITY;
}

// Runs one group of checks; an exception counts as a failure of the group
// instead of ending the test process.
template <class F>
void guarded(const char* label, F&& f) {
    try { f(); }
    catch (const std::exception& e) { check(false, std::string(label) + " threw: " + e.what()); }
    catch (...) { check(false, std::string(label) + " threw a non-std exception"); }
}

bool throws_invalid(const ODE_SIR& m, const AI4BayesCode::state_map& nd, bool single) {
    try { (void)m.predict_at(nd, single); }
    catch (const std::invalid_argument&) { return true; }
    catch (...) { return false; }
    return false;
}

}  // namespace

int main() {
    // Unbuffered, so the lines printed before a crash reach the log.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("=== test_fix_examples_ODE_SIR ===\n");
    const std::size_t n_obs = 15;
    arma::vec t_obs(n_obs);
    for (std::size_t k = 0; k < n_obs; ++k) t_obs[k] = static_cast<double>(k);
    const arma::vec y0{990.0, 10.0, 0.0};
    const arma::mat tr = AI4BayesCode::ode::rk45(sir_rhs, y0, t_obs,
                                                 arma::vec{0.6, 0.2}, 1e-9, 1e-9);
    std::mt19937_64 g(20260621ULL);
    std::normal_distribution<double> nd(0.0, 1.0);
    arma::vec I_obs(n_obs);
    for (std::size_t k = 0; k < n_obs; ++k) I_obs[k] = tr(k, 1) * std::exp(0.05 * nd(g));

    ODE_SIR m(990.0, 10.0, 0.0, t_obs, I_obs, 7, false);
    std::printf("C1 construction: max rel |I_traj - solution| = %.3g, z = %.3f\n",
                traj_error(m), yrep_z(m));
    check(traj_error(m) <= 1e-12, "C1 I_traj equals the solution at the initial theta");
    check(std::abs(yrep_z(m)) < 5.0, "C1 predict_at({}) I_rep centred on the solution");

    m.step(1);   // C2 starts from a state written by a sweep
    AI4BayesCode::state_map s;
    s["beta"]  = arma::vec{0.9};
    s["gamma"] = arma::vec{0.15};
    s["sigma"] = arma::vec{0.1};
    m.set_current(s);
    std::printf("C2 set_current(beta, gamma, sigma): max rel |I_traj - solution| = %.3g, "
                "z = %.3f\n", traj_error(m), yrep_z(m));
    check(traj_error(m) <= 1e-12, "C2 I_traj equals the solution at the loaded theta");
    check(std::abs(yrep_z(m)) < 5.0, "C2 predict_at({}) I_rep centred after set_current");

    m.step(5);
    const auto before = m.get_current();
    m.set_current(before);
    const auto after = m.get_current();
    check(same_bits(before.at("beta"), after.at("beta")) &&
          same_bits(before.at("gamma"), after.at("gamma")) &&
          same_bits(before.at("sigma"), after.at("sigma")) && traj_error(m) <= 1e-12,
          "C3 set_current(get_current()) leaves the state unchanged and I_traj exact");

    for (const double b : {1e5, std::numeric_limits<double>::infinity()}) {
        ODE_SIR m4(990.0, 10.0, 0.0, t_obs, I_obs, 7, false);
        m4.step(1);
        const auto& d = data_of(m4);
        const AI4BayesCode::composite_block& c = *(m4.*member_of(impl_tag{}));
        const auto values0 = d.values();
        const arma::vec block0 = c.child(0).current();
        AI4BayesCode::state_map s4;
        s4["beta"] = arma::vec{b};
        bool threw = false;
        try { m4.set_current(s4); } catch (const std::exception&) { threw = true; }
        bool ok = false;
        if (threw) {
            ok = d.values().size() == values0.size() &&
                 same_bits(block0, c.child(0).current());
            for (const auto& kv : values0)
                ok = ok && d.has(kv.first) && same_bits(kv.second, d.get(kv.first));
        } else {
            try { ok = traj_error(m4) <= 1e-12; } catch (const std::exception&) { ok = false; }
        }
        std::printf("C4 set_current(beta = %g): threw = %d, %s\n", b, threw ? 1 : 0,
                    threw ? (ok ? "state unchanged" : "state changed")
                          : (ok ? "I_traj exact" : "I_traj not the solution"));
        check(ok, std::string("C4 set_current(beta = ") + (std::isinf(b) ? "Inf" : "1e5") +
                  "): unchanged after the error, or I_traj exact");
    }

    {
        ODE_SIR me(990.0, 10.0, 0.0, t_obs, I_obs, 17, false);
        me.step(1);
        bool ok = true;
        for (const char* key : {"beta", "gamma", "sigma"}) {
            const auto before = me.get_current();
            AI4BayesCode::state_map se;
            se[key] = arma::vec();
            bool threw = false;
            try { me.set_current(se); } catch (const std::exception&) { threw = true; }
            const auto after = me.get_current();
            ok = ok && threw && after.size() == before.size();
            for (const auto& kv : before)
                ok = ok && after.count(kv.first) && same_bits(kv.second, after.at(kv.first));
        }
        check(ok, "C5 set_current with an empty beta, gamma or sigma throws and leaves get_current() unchanged");
    }

    // ---- predict_at with new t_obs / y0 -----------------------------------
    const int seed_p = 23;
    const int n_sweeps = 25;
    auto fitted = [&](bool keep) {
        auto mp = std::make_unique<ODE_SIR>(990.0, 10.0, 0.0, t_obs, I_obs, seed_p, keep);
        mp->step(n_sweeps);
        return mp;
    };
    guarded("C6", [&] {
        AI4BayesCode::state_map nd_t;  nd_t["t_obs"] = t_obs;
        AI4BayesCode::state_map nd_ty; nd_ty["t_obs"] = t_obs; nd_ty["y0"] = y0;
        const auto pa = fitted(true)->predict_at(AI4BayesCode::state_map{});
        const auto pb = fitted(true)->predict_at(nd_t);
        const auto pc = fitted(true)->predict_at(nd_ty);
        std::printf("C6 keys %zu, I_traj %llu x %llu\n", pa.size(),
                    (unsigned long long)pa.at("I_traj").n_rows,
                    (unsigned long long)pa.at("I_traj").n_cols);
        check(pa.size() == 2 && pa.count("I_traj") && pa.count("y_rep") &&
              pa.at("I_traj").n_rows == static_cast<arma::uword>(n_sweeps) &&
              same_maps(pa, pb) && same_maps(pa, pc),
              "C6 predict_at({}), ({t_obs}) and ({y0, t_obs}) at the training values agree bit for bit");
    });

    const arma::vec t_new{2.5, 7.0, 15.0, 21.0, 28.0};
    const arma::vec y0_new{800.0, 40.0, 160.0};
    guarded("C7", [&] {
        auto mp = fitted(true);
        const auto hist0 = mp->get_history();
        const auto cur0  = mp->get_current();
        const auto vals0 = data_of(*mp).values();
        AI4BayesCode::state_map nd; nd["t_obs"] = t_new; nd["y0"] = y0_new;
        const auto p = mp->predict_at(nd);
        std::mt19937_64 g = predict_stream(seed_p);
        const row_check rc = check_rows(p.at("I_traj"), p.at("y_rep"), hist0.at("theta"),
                                        hist0.at("sigma"), y0_new, t_new, g);
        std::printf("C7 rows %llu x %llu; I_traj mismatches %d, y_rep mismatches %d, "
                    "max rel err vs 1e-10 solve %.3g\n",
                    (unsigned long long)p.at("I_traj").n_rows,
                    (unsigned long long)p.at("I_traj").n_cols,
                    rc.I_mismatch, rc.y_mismatch, rc.rel_err);
        check(p.size() == 2 && p.at("I_traj").n_rows == static_cast<arma::uword>(n_sweeps) &&
              p.at("I_traj").n_cols == t_new.n_elem && p.at("y_rep").n_rows == p.at("I_traj").n_rows &&
              p.at("y_rep").n_cols == t_new.n_elem,
              "C7 one row per stored draw, one column per new time");
        check(rc.I_mismatch == 0 && rc.rel_err < 1e-5,
              "C7 I_traj row d = solution from the new y0 at time 0 with theta_d");
        check(rc.y_mismatch == 0, "C7 y_rep row d = I_traj * exp(sigma_d * z), z from predict_rng_");
        bool same_vals = data_of(*mp).values().size() == vals0.size();
        for (const auto& kv : vals0)
            same_vals = same_vals && data_of(*mp).has(kv.first) &&
                        same_bits(kv.second, data_of(*mp).get(kv.first));
        const auto cur1 = mp->get_current();
        bool same_cur = cur1.size() == cur0.size();
        for (const auto& kv : cur0)
            same_cur = same_cur && cur1.count(kv.first) && same_bits(kv.second, cur1.at(kv.first));
        check(same_maps(hist0, mp->get_history()) && same_cur && same_vals,
              "C7 predict_at at new data leaves the history, the current state and the shared data unchanged");
    });
    guarded("C8", [&] {
        auto mp = fitted(true);
        AI4BayesCode::state_map nd; nd["t_obs"] = arma::regspace(0.0, 1.0, 9.0);
        const auto p = mp->predict_at(nd);
        const arma::mat& I = p.at("I_traj");
        bool ok = I.n_rows == static_cast<arma::uword>(n_sweeps) && I.n_cols == 10;
        for (arma::uword r = 0; ok && r < I.n_rows; ++r) ok = I(r, 0) == 10.0;
        check(ok, "C8 at times 0..9 the first column of I_traj is I0 = 10");
    });
    guarded("C9", [&] {
        AI4BayesCode::state_map nd; nd["t_obs"] = t_new;
        for (const bool keep : {false, true}) {
            auto mp = fitted(keep);
            const auto p = mp->predict_at(nd, /*last_draw_only=*/true);
            const auto cur = mp->get_current();
            arma::mat theta_cur(1, 2), sigma_cur(1, 1);
            theta_cur(0, 0) = cur.at("beta")[0];
            theta_cur(0, 1) = cur.at("gamma")[0];
            sigma_cur(0, 0) = cur.at("sigma")[0];
            std::mt19937_64 g = predict_stream(seed_p);
            const row_check rc = check_rows(p.at("I_traj"), p.at("y_rep"), theta_cur, sigma_cur,
                                            y0, t_new, g);
            check(p.size() == 2 && p.at("I_traj").n_rows == 1 && p.at("y_rep").n_rows == 1 &&
                  rc.I_mismatch == 0 && rc.y_mismatch == 0 && rc.rel_err < 1e-5,
                  std::string("C9 single-draw path (keep_history = ") + (keep ? "true" : "false") +
                  ", last_draw_only): one row from the current theta and sigma, training y0");
            if (!keep) {
                const auto p1 = mp->predict_at(nd);   // keep_history = false: same path
                check(p1.size() == 2 && p1.at("I_traj").n_rows == 1 &&
                      same_bits_mat(p1.at("I_traj"), p.at("I_traj")),
                      "C9 keep_history = false: predict_at(nd) takes the single-draw path");
            }
        }
    });
    guarded("C10", [&] {
        auto mp = fitted(true);
        mp->clear_history();
        AI4BayesCode::state_map nd; nd["t_obs"] = t_new;
        const auto p = mp->predict_at(nd);
        const auto cur = mp->get_current();
        const arma::vec th{cur.at("beta")[0], cur.at("gamma")[0]};
        check(p.at("I_traj").n_rows == 1 && p.at("y_rep").n_rows == 1 &&
              same_bits(p.at("I_traj").row(0).t(), sir_I_at(y0, t_new, th, 1.0)),
              "C10 after clear_history(): one row from the current state");
    });
    guarded("C11", [&] {
        auto mp = fitted(true);
        const auto hist0 = mp->get_history();
        const double inf = std::numeric_limits<double>::infinity();
        const double nan = std::numeric_limits<double>::quiet_NaN();
        std::vector<std::pair<std::string, arma::vec>> bad = {
            {"t_obs", arma::vec()}, {"t_obs", arma::vec{-1.0, 2.0}},
            {"t_obs", arma::vec{1.0, nan}}, {"t_obs", arma::vec{1.0, inf}},
            {"t_obs", arma::vec{3.0, 2.0}}, {"t_obs", arma::vec{1.0, 1.0}},
            {"y0", arma::vec{990.0, 10.0}}, {"y0", arma::vec{990.0, 10.0, 0.0, 1.0}},
            {"y0", arma::vec{990.0, -10.0, 0.0}}, {"y0", arma::vec{nan, 10.0, 0.0}},
            {"y0", arma::vec{inf, 10.0, 0.0}}, {"y0", arma::vec{0.0, 0.0, 0.0}},
            {"y0", arma::vec{1e308, 1e308, 0.0}},
            {"I_obs", I_obs}, {"theta", arma::vec{0.5, 0.2}}, {"t", t_new}};
        int n_ok = 0;
        for (const auto& b : bad) {
            AI4BayesCode::state_map nd; nd[b.first] = b.second;
            const bool h = throws_invalid(*mp, nd, false);
            const bool s = throws_invalid(*mp, nd, true);
            if (h && s) ++n_ok;
            else std::printf("  C11 not rejected: key %s (history %d, single %d)\n",
                             b.first.c_str(), h ? 1 : 0, s ? 1 : 0);
        }
        const auto p = mp->predict_at(AI4BayesCode::state_map{});
        check(n_ok == static_cast<int>(bad.size()) && same_maps(hist0, mp->get_history()) &&
              p.at("I_traj").n_rows == static_cast<arma::uword>(n_sweeps),
              "C11 invalid new data throws std::invalid_argument; history unchanged, predict_at({}) gives one row per stored draw");
    });

    guarded("C12", [&] {
        auto mp = fitted(true);
        const arma::mat theta = mp->get_history().at("theta");
        const arma::vec t_far{30.0, 60.0, 90.0, 120.0};
        const arma::vec t_all = arma::regspace(1.0, 1.0, 120.0);
        const arma::uvec far_cols{29, 59, 89, 119};
        const arma::vec y0_frac{0.99, 0.01, 0.0};
        double e_grid = 0.0, e_rk4 = 0.0, e_scale = 0.0;
        bool train_same = true, rows_ok = true;
        arma::mat far_count;
        for (const arma::vec& y0_k : {y0, y0_frac}) {
            AI4BayesCode::state_map nd_far; nd_far["t_obs"] = t_far; nd_far["y0"] = y0_k;
            AI4BayesCode::state_map nd_all; nd_all["t_obs"] = t_all; nd_all["y0"] = y0_k;
            AI4BayesCode::state_map nd_tr;  nd_tr["y0"] = y0_k;
            const arma::mat I_far = mp->predict_at(nd_far).at("I_traj");
            const arma::mat I_all = mp->predict_at(nd_all).at("I_traj");
            const arma::mat I_tr  = mp->predict_at(nd_tr).at("I_traj");
            rows_ok = rows_ok && I_far.n_rows == theta.n_rows && I_all.n_rows == theta.n_rows &&
                      I_tr.n_rows == theta.n_rows;
            for (arma::uword d = 0; rows_ok && d < theta.n_rows; ++d) {
                const arma::vec th = theta.row(d).t();
                const arma::vec ref = rk4_I(y0_k, t_far, th);
                const arma::vec far = I_far.row(d).t();
                const arma::vec all = arma::vec(I_all.row(d).t()).elem(far_cols);
                e_grid = std::max(e_grid, max_rel(far, all));
                e_rk4  = std::max(e_rk4, std::max(max_rel(far, ref), max_rel(all, ref)));
                const arma::vec plain = AI4BayesCode::ode::rk45(sir_rhs, y0_k, t_obs, th,
                                                                1e-6, 1e-6).col(1);
                train_same = train_same && same_bits(arma::vec(I_tr.row(d).t()), plain);
            }
            if (far_count.n_elem == 0) far_count = I_far;
            else e_scale = std::max(e_scale, arma::mat(arma::abs(1000.0 * I_far - far_count) /
                                                       arma::abs(far_count)).max());
        }
        std::printf("C12 max rel: alone vs within 1..120 %.3g, vs RK4 %.3g, fraction*1000 vs "
                    "count %.3g; training times bit-identical %d\n",
                    e_grid, e_rk4, e_scale, train_same ? 1 : 0);
        check(rows_ok && e_grid < 1e-4 && e_rk4 < 1e-4 && e_scale < 1e-4,
              "C12 I_traj at t = 30..120 does not depend on the other requested times or "
              "the scale of y0, and matches RK4");
        check(rows_ok && train_same,
              "C12 at the training times I_traj equals the uncapped training solve bit for bit");
    });

    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "ALL PASSED", g_fail,
                g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}

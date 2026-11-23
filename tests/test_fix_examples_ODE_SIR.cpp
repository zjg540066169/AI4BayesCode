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
//  The solution is recomputed in this test with the example's own sir_rhs
//  and solver settings (relative tolerance 1e-12). Centred: the mean of
//  log(I_rep / I_traj) / sigma over the 15 times lies within
//  5 / sqrt(15) of 0.
// ============================================================================
#include <armadillo>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <string>

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

    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "ALL PASSED", g_fail,
                g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}

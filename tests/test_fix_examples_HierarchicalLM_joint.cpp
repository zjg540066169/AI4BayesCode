// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_examples_HierarchicalLM_joint.cpp
//
//  The derived keys "mu_fixed" = alpha + X beta and "u" = tau * z_u of
//  examples/HierarchicalLM_joint.cpp, from which predict_at({}) draws
//  y_rep = mu_fixed + u[g] + sigma * N(0, 1).
//    C1  after construction, both equal the values at the initial state;
//    C2  after one sweep and set_current of alpha, beta and tau, both equal
//        the values at the loaded state, and y_rep is centred on them;
//    C3  set_current(get_current()) after 5 more sweeps returns the same
//        get_current() values, with mu_fixed and u exact.
//  The values are recomputed with loops in this test (relative tolerance
//  1e-12). Centred: the mean of (y_rep - mu_fixed - u[g]) / sigma lies
//  within 5 / sqrt(N) of 0.
//  predict_at(new_data) with g_idx (0-based):
//    C4  a negative, non-integer or NaN entry, a length other than the rows
//        of X, or (without X) other than N throws; g_idx of length N without
//        X, and entries >= G with X, are accepted;
//    C5  history mode with entries >= G: y_rep of draw d equals
//        alpha_d + X beta_d + u + sigma_d * e bit for bit, where u =
//        tau_d * z_u_d[g] for g < G and, for g >= G, u = tau_d * z drawn at
//        the first row of g in draw d and reused for its later rows; z and e
//        come from an engine seeded as predict_rng_, in row order;
//    C6  the single-draw path follows the same rule at the current state
//        (mu_fixed + u + sigma * e, u = tau * z once per new group).
// ============================================================================
#include <armadillo>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>

#include "AI4BayesCode/composite_block.hpp"

#define main hierarchical_lm_joint_demo_main
#include "../examples/HierarchicalLM_joint.cpp"
#undef main

// An explicit instantiation may name a private member: this gives the test
// read access to the wrapper's composite_block.
template <typename Tag, typename Tag::type Member>
struct expose_member {
    friend typename Tag::type member_of(Tag) { return Member; }
};
struct impl_tag {
    using type = std::unique_ptr<AI4BayesCode::composite_block> HierarchicalLM_joint::*;
    friend type member_of(impl_tag);
};
template struct expose_member<impl_tag, &HierarchicalLM_joint::impl_>;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

const AI4BayesCode::shared_data_t& data_of(const HierarchicalLM_joint& m) {
    return (m.*member_of(impl_tag{}))->data();
}

double rel_err(const arma::vec& a, const arma::vec& b) {
    if (a.n_elem != b.n_elem) return INFINITY;
    double e = 0.0;
    for (std::size_t i = 0; i < a.n_elem; ++i)
        e = std::max(e, std::abs(a[i] - b[i]) / std::max(1.0, std::abs(b[i])));
    return e;
}

// alpha + X beta from the current alpha, beta and X.
arma::vec mu_ref(const HierarchicalLM_joint& m) {
    const auto& d = data_of(m);
    const double alpha = d.get("alpha")[0];
    const arma::vec& beta = d.get("beta");
    const arma::vec& X = d.get("X");
    const std::size_t p = beta.n_elem, N = X.n_elem / p;
    arma::vec mu(N);
    for (std::size_t n = 0; n < N; ++n) {
        double xb = 0.0;
        for (std::size_t k = 0; k < p; ++k) xb += X[n + k * N] * beta[k];
        mu[n] = alpha + xb;
    }
    return mu;
}

arma::vec u_ref(const HierarchicalLM_joint& m) {
    const auto& d = data_of(m);
    return d.get("tau")[0] * d.get("z_u");
}

double derived_error(const HierarchicalLM_joint& m) {
    const auto& d = data_of(m);
    return std::max(rel_err(d.get("mu_fixed"), mu_ref(m)), rel_err(d.get("u"), u_ref(m)));
}

double yrep_z(const HierarchicalLM_joint& m) {
    const arma::mat yr = m.predict_at(AI4BayesCode::state_map{}).at("y_rep");
    const arma::vec mu = mu_ref(m), u = u_ref(m);
    const arma::vec& g = data_of(m).get("g_idx");
    const double sigma = data_of(m).get("sigma")[0];
    double s = 0.0;
    for (std::size_t n = 0; n < mu.n_elem; ++n)
        s += (yr[n] - mu[n] - u[static_cast<std::size_t>(g[n])]) / sigma;
    return s / std::sqrt(static_cast<double>(mu.n_elem));
}

bool same_bits(const arma::vec& a, const arma::vec& b) {
    return a.n_elem == b.n_elem &&
           std::memcmp(a.memptr(), b.memptr(), a.n_elem * sizeof(double)) == 0;
}

}  // namespace

int main() {
    std::printf("=== test_fix_examples_HierarchicalLM_joint ===\n");
    const std::size_t N = 200, p = 2;
    const int G = 8;
    std::mt19937_64 gen(17u);
    std::normal_distribution<double> nd(0.0, 1.0);
    arma::mat X(N, p);
    arma::vec gidx(N), y(N);
    arma::vec u_true(G);
    for (int g = 0; g < G; ++g) u_true[g] = 0.7 * nd(gen);
    for (std::size_t n = 0; n < N; ++n) {
        X(n, 0) = nd(gen); X(n, 1) = nd(gen);
        gidx[n] = static_cast<double>(1 + n % G);
        y[n] = 1.0 + 0.8 * X(n, 0) - 0.5 * X(n, 1) + u_true[n % G] + 0.4 * nd(gen);
    }

    HierarchicalLM_joint m(y, X, gidx, G, 1.0, 1.0, 7, false);
    std::printf("C1 construction: max rel error of mu_fixed, u = %.3g\n", derived_error(m));
    check(derived_error(m) <= 1e-12, "C1 mu_fixed and u equal the values at the initial state");

    m.step(1);
    AI4BayesCode::state_map s;
    s["alpha"] = arma::vec{-2.0};
    s["beta"]  = arma::vec{3.0, -4.0};
    s["tau"]   = arma::vec{2.5};
    m.set_current(s);
    std::printf("C2 set_current(alpha, beta, tau): max rel error of mu_fixed, u = %.3g, "
                "y_rep z = %.3f\n", derived_error(m), yrep_z(m));
    check(derived_error(m) <= 1e-12, "C2 mu_fixed and u equal the values at the loaded state");
    check(std::abs(yrep_z(m)) < 5.0, "C2 predict_at({}) y_rep centred on the loaded state");

    m.step(5);
    const auto before = m.get_current();
    m.set_current(before);
    const auto after = m.get_current();
    bool same = true;
    for (const char* k : {"alpha", "beta", "z_u", "u", "sigma", "tau"})
        same = same && same_bits(before.at(k), after.at(k));
    check(same && derived_error(m) <= 1e-12,
          "C3 set_current(get_current()) leaves the state unchanged and mu_fixed, u exact");

    const std::uint64_t pred_xor = 0x9E3779B97F4A7C15ULL;   // predict_rng_ seed
    const arma::mat X5 = X.rows(0, 4);
    const arma::vec Xf5 = arma::vectorise(X5);
    const arma::vec g5{0.0, static_cast<double>(G), static_cast<double>(G + 1),
                       static_cast<double>(G), 3.0};
    AI4BayesCode::state_map nd5;
    nd5["X"] = Xf5;
    nd5["g_idx"] = g5;

    // C4
    {
        HierarchicalLM_joint mv(y, X, gidx, G, 1.0, 1.0, 19, true);
        mv.step(3);
        auto throws = [&](const AI4BayesCode::state_map& nd) {
            try { (void)mv.predict_at(nd); } catch (const std::exception& e) {
                std::printf("    error: %s\n", e.what());
                return true;
            }
            return false;
        };
        auto with_X = [&](const arma::vec& g) {
            AI4BayesCode::state_map nd;
            nd["X"] = arma::vectorise(X.rows(0, 2));
            nd["g_idx"] = g;
            return nd;
        };
        AI4BayesCode::state_map short_no_X, full_no_X;
        short_no_X["g_idx"] = arma::vec{0.0, 1.0, 2.0};
        full_no_X["g_idx"] = gidx - 1.0;
        const bool rejects = throws(with_X(arma::vec{0.0, -1.0, 2.0})) &&
                             throws(with_X(arma::vec{0.0, 1.5, 2.0})) &&
                             throws(with_X(arma::vec{0.0, NAN, 2.0})) &&
                             throws(with_X(arma::vec{0.0, 1.0})) && throws(short_no_X);
        const bool accepts = !throws(full_no_X) && !throws(nd5);
        check(rejects && accepts,
              "C4 g_idx: negative, non-integer, NaN and wrong lengths throw; N entries "
              "without X and entries >= G are accepted");
    }

    // C5: history mode, groups >= G.
    {
        HierarchicalLM_joint hh(y, X, gidx, G, 1.0, 1.0, 21, true);
        hh.step(12);
        const auto H = hh.get_history();
        const auto P = hh.predict_at(nd5);
        const arma::mat& A = H.at("alpha");
        const arma::mat& B = H.at("beta");
        const arma::mat& Zu = H.at("z_u");
        const arma::mat& T = H.at("tau");
        const arma::mat& S = H.at("sigma");
        const arma::mat& YR = P.at("y_rep");
        bool bits = YR.n_rows == A.n_rows && YR.n_cols == 5 && A.n_rows == 12;
        std::mt19937_64 r(static_cast<std::uint64_t>(21) ^ pred_xor);
        std::normal_distribution<double> z01(0.0, 1.0);
        std::unordered_map<std::size_t, double> u_new;
        for (std::size_t d = 0; bits && d < A.n_rows; ++d) {
            u_new.clear();
            for (std::size_t i = 0; i < 5; ++i) {
                double xb = 0.0;
                for (std::size_t j = 0; j < p; ++j) xb += Xf5[i + j * 5] * B(d, j);
                const std::size_t gi = static_cast<std::size_t>(g5[i]);
                double u;
                if (gi < static_cast<std::size_t>(G)) {
                    u = T(d, 0) * Zu(d, gi);
                } else {
                    auto it = u_new.find(gi);
                    if (it == u_new.end()) it = u_new.emplace(gi, T(d, 0) * z01(r)).first;
                    u = it->second;
                }
                const double mu = A(d, 0) + xb + u;
                const double yr = mu + S(d, 0) * z01(r);
                const double out = YR(d, i);
                bits = bits && std::memcmp(&yr, &out, sizeof(double)) == 0;
            }
        }
        std::printf("C5 history predict_at with g_idx = {0, G, G+1, G, 3}: %u x %u, replay %s\n",
                    static_cast<unsigned>(YR.n_rows), static_cast<unsigned>(YR.n_cols),
                    bits ? "bit-identical" : "DIFFERS");
        check(bits, "C5 history mode draws u ~ N(0, tau_d^2) once per new group and draw");
    }

    // C6: single-draw path, groups >= G.
    {
        HierarchicalLM_joint ms(y, X, gidx, G, 1.0, 1.0, 23, false);
        ms.step(4);
        const auto cur = ms.get_current();
        const auto P = ms.predict_at(nd5);
        const arma::mat& MF = P.at("mu_fixed");
        const arma::mat& YR = P.at("y_rep");
        bool bits = P.size() == 2 && MF.n_rows == 1 && YR.n_rows == 1 && YR.n_cols == 5;
        std::mt19937_64 r(static_cast<std::uint64_t>(23) ^ pred_xor);
        std::normal_distribution<double> z01(0.0, 1.0);
        std::unordered_map<std::size_t, double> u_new;
        const double tau = cur.at("tau")[0], sigma = cur.at("sigma")[0];
        for (std::size_t i = 0; bits && i < 5; ++i) {
            const std::size_t gi = static_cast<std::size_t>(g5[i]);
            double u;
            if (gi < static_cast<std::size_t>(G)) {
                u = cur.at("u")[gi];
            } else {
                auto it = u_new.find(gi);
                if (it == u_new.end()) it = u_new.emplace(gi, tau * z01(r)).first;
                u = it->second;
            }
            const double yr = MF(0, i) + u + sigma * z01(r);
            const double out = YR(0, i);
            bits = bits && std::memcmp(&yr, &out, sizeof(double)) == 0;
        }
        std::printf("C6 single-draw predict_at with g_idx = {0, G, G+1, G, 3}: replay %s\n",
                    bits ? "bit-identical" : "DIFFERS");
        check(bits, "C6 the single-draw path draws u ~ N(0, tau^2) once per new group");
    }

    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "ALL PASSED", g_fail,
                g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}

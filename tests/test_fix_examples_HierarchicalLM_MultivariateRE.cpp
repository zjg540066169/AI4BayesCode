// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_examples_HierarchicalLM_MultivariateRE.cpp
//
//  The derived keys of examples/HierarchicalLM_MultivariateRE.cpp right
//  after construction, before any sweep:
//    C1  "mu_fixed" equals X beta, "tau" equals exp(log_tau), "sigma" equals
//        exp(log_sigma) and "u" equals diag(tau) L z, all at the initial
//        state;
//    C2  y_rep from predict_at({}) is centred on mu_fixed + Z u[g]: the mean
//        of (y_rep - mu_fixed - Z u[g]) / sigma lies within 5 / sqrt(N) of 0.
//  The values are recomputed with loops in this test (relative tolerance
//  1e-12).
//  set_current (beta, z, tau, L and sigma are sub-parameters of the joint
//  block, tau and sigma through log_tau and log_sigma):
//    C3  after one sweep, set_current of beta, z, tau, L and sigma: get_current
//        reports beta and z bit for bit and tau, sigma and L within 1e-12,
//        and mu_fixed = X beta, u = diag(tau) L z at the loaded values;
//    C4  the next two sweeps equal those of a twin loaded through
//        composite_block::set_current with [beta, z, log tau, log sigma, L];
//    C5  a valid beta with an L whose rows do not have unit length, with an
//        L that has a non-zero entry above the diagonal, or with tau = -1
//        throws and leaves every shared value and the block's current value
//        unchanged (bit for bit);
//    C6  set_current(get_current()) leaves get_current() unchanged up to
//        rounding (1e-12).
//  predict_at(new_data) with group_idx (1-based):
//    C7  an entry < 1, a non-integer or NaN entry, a length other than the
//        rows of X, or (without X) other than N throws; group_idx of length
//        N without X, and labels > J with X, are accepted;
//    C8  history mode with labels > J: y_rep of draw d equals
//        X beta_d + Z u + sigma_d * e bit for bit, where u = u_d[:, g] for a
//        training group and, for a label > J, u = diag(tau_d) L_d z with
//        z ~ N(0, I_2) drawn at the first row of that label in draw d and
//        reused for its later rows; z and e come from an engine seeded as
//        predict_rng_, in row order.
// ============================================================================
#include <armadillo>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "AI4BayesCode/composite_block.hpp"

#define main hierarchical_lm_mvre_demo_main
#include "../examples/HierarchicalLM_MultivariateRE.cpp"
#undef main

// An explicit instantiation may name a private member: this gives the test
// read access to the wrapper's composite_block.
template <typename Tag, typename Tag::type Member>
struct expose_member {
    friend typename Tag::type member_of(Tag) { return Member; }
};
struct impl_tag {
    using type = std::unique_ptr<AI4BayesCode::composite_block>
        HierarchicalLM_MultivariateRE::*;
    friend type member_of(impl_tag);
};
template struct expose_member<impl_tag, &HierarchicalLM_MultivariateRE::impl_>;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

double rel_err(const arma::vec& a, const arma::vec& b) {
    if (a.n_elem != b.n_elem) return INFINITY;
    double e = 0.0;
    for (std::size_t i = 0; i < a.n_elem; ++i)
        e = std::max(e, std::abs(a[i] - b[i]) / std::max(1.0, std::abs(b[i])));
    return e;
}

bool same_bits(const arma::vec& a, const arma::vec& b) {
    return a.n_elem == b.n_elem &&
           std::memcmp(a.memptr(), b.memptr(), a.n_elem * sizeof(double)) == 0;
}

AI4BayesCode::composite_block& impl(const HierarchicalLM_MultivariateRE& m) {
    return *(m.*member_of(impl_tag{}));
}

// set_current(s) throws, and every shared value and the block's current
// value are unchanged.
bool throws_unchanged(HierarchicalLM_MultivariateRE& m, const AI4BayesCode::state_map& s) {
    const std::map<std::string, arma::vec> shared0(impl(m).data().values().begin(),
                                                   impl(m).data().values().end());
    const arma::vec blk0 = impl(m).child(0).current();
    bool threw = false;
    try { m.set_current(s); } catch (const std::exception&) { threw = true; }
    bool same = threw && same_bits(impl(m).child(0).current(), blk0) &&
                impl(m).data().values().size() == shared0.size();
    for (const auto& kv : shared0) same = same && same_bits(impl(m).data().get(kv.first), kv.second);
    return same;
}

}  // namespace

int main() {
    std::printf("=== test_fix_examples_HierarchicalLM_MultivariateRE ===\n");
    const std::size_t J = 12, per = 15, N = J * per, p = 3;
    std::mt19937_64 gen(23u);
    std::normal_distribution<double> nd(0.0, 1.0);
    arma::mat X(N, p, arma::fill::ones);
    arma::vec group(N), y(N);
    for (std::size_t i = 0; i < N; ++i) {
        X(i, 1) = nd(gen); X(i, 2) = nd(gen);
        group[i] = static_cast<double>(i / per) + 1.0;
        y[i] = 5.0 + 1.5 * X(i, 1) - 2.0 * X(i, 2) + 0.5 * nd(gen);
    }

    HierarchicalLM_MultivariateRE m(y, X, group, 7, false);
    const auto& d = (m.*member_of(impl_tag{}))->data();

    // C1
    const arma::vec& beta = d.get("beta");
    arma::vec mu(N);
    for (std::size_t i = 0; i < N; ++i) {
        double s = 0.0;
        for (std::size_t k = 0; k < p; ++k) s += X(i, k) * beta[k];
        mu[i] = s;
    }
    const arma::vec tau = arma::exp(d.get("log_tau"));
    const double sigma = std::exp(d.get("log_sigma")[0]);
    const arma::mat L(d.get("R_chol").memptr(), D, D);
    const arma::mat Zm(d.get("z_flat").memptr(), D, J);
    arma::mat Um = L * Zm;
    for (std::size_t k = 0; k < D; ++k) Um.row(k) *= tau[k];
    const arma::vec u = arma::vectorise(Um);
    const double e_mu = rel_err(d.get("mu_fixed"), mu);
    const double e_tau = rel_err(d.get("tau"), tau);
    const double e_sig = rel_err(d.get("sigma"), arma::vec{sigma});
    const double e_u = rel_err(d.get("u"), u);
    std::printf("C1 construction: rel error mu_fixed %.3g, tau %.3g, sigma %.3g, u %.3g\n",
                e_mu, e_tau, e_sig, e_u);
    check(e_mu <= 1e-12 && e_tau <= 1e-12 && e_sig <= 1e-12 && e_u <= 1e-12,
          "C1 mu_fixed, tau, sigma and u equal the values at the initial state");

    // C2
    const arma::mat yr = m.predict_at(AI4BayesCode::state_map{}).at("y_rep");
    double z = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        const std::size_t g = static_cast<std::size_t>(group[i]) - 1;
        double re = 0.0;
        for (std::size_t k = 0; k < D; ++k) re += X(i, k) * u[k + g * D];
        z += (yr[i] - mu[i] - re) / sigma;
    }
    z /= std::sqrt(static_cast<double>(N));
    std::printf("C2 predict_at({}) before the first sweep: y_rep z = %.3f\n", z);
    check(std::abs(z) < 5.0, "C2 y_rep is centred on mu_fixed + Z u[g] of the initial state");

    // C3
    HierarchicalLM_MultivariateRE m3(y, X, group, 7, false), m4(y, X, group, 7, false);
    m3.step(1);
    m4.step(1);
    const arma::vec b3{4.0, 1.0, -1.0};
    arma::vec z3(D * J);
    for (std::size_t i = 0; i < D * J; ++i) z3[i] = 0.1 * static_cast<double>(i % 7) - 0.3;
    const arma::vec t3{0.8, 0.4};
    const double s3v = 0.6;
    const arma::vec L3{1.0, 0.6, 0.0, 0.8};   // col-major: L(1, 0) = 0.6, L(1, 1) = 0.8
    AI4BayesCode::state_map s3;
    s3["beta"] = b3;
    s3["z"] = z3;
    s3["tau"] = t3;
    s3["L"] = L3;
    s3["sigma"] = arma::vec{s3v};
    m3.set_current(s3);
    const auto g3 = m3.get_current();
    const auto& d3 = impl(m3).data();
    const arma::vec mu3 = X * b3;
    const arma::mat Lm3(d3.get("R_chol").memptr(), D, D);
    const arma::mat Zm3(d3.get("z_flat").memptr(), D, J);
    arma::mat Um3 = Lm3 * Zm3;
    for (std::size_t k = 0; k < D; ++k) Um3.row(k) *= d3.get("tau")[k];
    const double e_t = rel_err(g3.at("tau"), t3), e_s = rel_err(g3.at("sigma"), arma::vec{s3v}),
                 e_L = rel_err(g3.at("L"), L3), e_mu3 = rel_err(d3.get("mu_fixed"), mu3),
                 e_u3 = rel_err(d3.get("u"), arma::vectorise(Um3));
    std::printf("C3 set_current: rel error tau %.3g, sigma %.3g, L %.3g, mu_fixed %.3g, u %.3g\n",
                e_t, e_s, e_L, e_mu3, e_u3);
    check(same_bits(g3.at("beta"), b3) && same_bits(g3.at("z"), z3) && e_t <= 1e-12 &&
          e_s <= 1e-12 && e_L <= 1e-12 && e_mu3 <= 1e-12 && e_u3 <= 1e-12,
          "C3 the loaded values are reported and mu_fixed, u are recomputed");

    // C4
    arma::vec cat4(p + D * J + D + 1 + D * D);
    cat4.subvec(0, p - 1) = b3;
    cat4.subvec(p, p + D * J - 1) = z3;
    for (std::size_t k = 0; k < D; ++k) cat4[p + D * J + k] = std::log(t3[k]);
    cat4[p + D * J + D] = std::log(s3v);
    cat4.subvec(p + D * J + D + 1, cat4.n_elem - 1) = L3;
    impl(m4).set_current(cat4);
    m3.step(2);
    m4.step(2);
    const auto a4 = m3.get_current(), b4 = m4.get_current();
    bool same4 = a4.size() == b4.size();
    for (const auto& kv : a4) same4 = same4 && same_bits(kv.second, b4.at(kv.first));
    check(same4, "C4 the sweeps after set_current equal those of a twin loaded through the composite");

    // C5
    AI4BayesCode::state_map s5a, s5b, s5c;
    s5a["beta"] = b3 + 1.0;
    s5a["L"] = arma::vec{1.0, 0.6, 0.0, 0.6};
    s5b["beta"] = b3 + 1.0;
    s5b["L"] = arma::vec{1.0, 0.6, 0.1, 0.8};
    s5c["beta"] = b3 + 1.0;
    s5c["tau"] = arma::vec{-1.0, 1.0};
    const bool c5a = throws_unchanged(m3, s5a), c5b = throws_unchanged(m3, s5b),
               c5c = throws_unchanged(m3, s5c);
    std::printf("C5 throws and changes nothing: rows of L not unit %d, L not lower-triangular %d, "
                "tau -1 %d\n", c5a ? 1 : 0, c5b ? 1 : 0, c5c ? 1 : 0);
    check(c5a && c5b && c5c, "C5 a call with an invalid key throws and changes nothing");

    // C6
    const auto c6 = m3.get_current();
    m3.set_current(c6);
    const auto a6 = m3.get_current();
    double e6 = 0.0;
    for (const auto& kv : c6) e6 = std::max(e6, rel_err(a6.at(kv.first), kv.second));
    std::printf("C6 set_current(get_current()): max rel change %.3g\n", e6);
    check(e6 <= 1e-12, "C6 set_current(get_current()) leaves get_current() unchanged up to rounding");

    const std::uint64_t pred_xor = 0x9E3779B97F4A7C15ULL;   // predict_rng_ seed
    const arma::mat X5 = X.rows(0, 4);
    const double Jd = static_cast<double>(J);
    const arma::vec g5{1.0, Jd + 1.0, Jd + 2.0, Jd + 1.0, 2.0};
    AI4BayesCode::state_map nd5;
    nd5["X"] = arma::vectorise(X5);
    nd5["group_idx"] = g5;

    // C7
    {
        HierarchicalLM_MultivariateRE mv(y, X, group, 31, false);
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
            nd["group_idx"] = g;
            return nd;
        };
        AI4BayesCode::state_map short_no_X, full_no_X;
        short_no_X["group_idx"] = arma::vec{1.0, 2.0, 3.0};
        full_no_X["group_idx"] = group;
        const bool rejects = throws(with_X(arma::vec{1.0, 0.0, 2.0})) &&
                             throws(with_X(arma::vec{1.0, -1.0, 2.0})) &&
                             throws(with_X(arma::vec{1.0, 1.5, 2.0})) &&
                             throws(with_X(arma::vec{1.0, NAN, 2.0})) &&
                             throws(with_X(arma::vec{1.0, 2.0})) && throws(short_no_X);
        const bool accepts = !throws(full_no_X) && !throws(nd5);
        check(rejects && accepts,
              "C7 group_idx: < 1, non-integer, NaN and wrong lengths throw; N entries "
              "without X and labels > J are accepted");
    }

    // C8: history mode, labels > J.
    {
        HierarchicalLM_MultivariateRE mh(y, X, group, 33, true);
        mh.step(6);
        const auto H = mh.get_history();
        const auto P = mh.predict_at(nd5);
        const arma::mat& B = H.at("beta");
        const arma::mat& Zf = H.at("z_flat");
        const arma::mat& LT = H.at("log_tau");
        const arma::mat& LS = H.at("log_sigma");
        const arma::mat& RC = H.at("R_chol");
        const arma::mat& YR = P.at("y_rep");
        bool bits = YR.n_rows == B.n_rows && YR.n_cols == 5 && B.n_rows == 6;
        std::mt19937_64 r(static_cast<std::uint64_t>(33) ^ pred_xor);
        std::normal_distribution<double> z01(0.0, 1.0);
        std::unordered_map<std::size_t, std::array<double, D>> u_new;
        for (std::size_t d = 0; bits && d < B.n_rows; ++d) {
            arma::vec tau_d(D);
            for (std::size_t k = 0; k < D; ++k) tau_d[k] = std::exp(LT(d, k));
            const double sigma_d = std::exp(LS(d, 0));
            arma::vec L_flat = RC.row(d).t();
            arma::vec z_d = Zf.row(d).t();
            arma::mat L_d(L_flat.memptr(), D, D, false, true);
            arma::mat z_mat(z_d.memptr(), D, J, false, true);
            arma::mat u_mat = L_d * z_mat;
            for (std::size_t k = 0; k < D; ++k) u_mat.row(k) *= tau_d[k];
            u_new.clear();
            for (std::size_t i = 0; i < 5; ++i) {
                double mu = 0.0;
                for (std::size_t k = 0; k < p; ++k) mu += X5(i, k) * B(d, k);
                const std::size_t g = static_cast<std::size_t>(g5[i]) - 1u;
                if (g < J) {
                    for (std::size_t k = 0; k < D; ++k) mu += X5(i, k) * u_mat(k, g);
                } else {
                    auto it = u_new.find(g);
                    if (it == u_new.end()) {
                        arma::vec zg(D);
                        for (std::size_t k = 0; k < D; ++k) zg[k] = z01(r);
                        arma::vec ug = L_d * zg;
                        std::array<double, D> ua;
                        for (std::size_t k = 0; k < D; ++k) ua[k] = tau_d[k] * ug[k];
                        it = u_new.emplace(g, ua).first;
                    }
                    for (std::size_t k = 0; k < D; ++k) mu += X5(i, k) * it->second[k];
                }
                const double yr = mu + sigma_d * z01(r);
                const double out = YR(d, i);
                bits = bits && std::memcmp(&yr, &out, sizeof(double)) == 0;
            }
        }
        std::printf("C8 history predict_at with group_idx = {1, J+1, J+2, J+1, 2}: %u x %u, "
                    "replay %s\n", static_cast<unsigned>(YR.n_rows),
                    static_cast<unsigned>(YR.n_cols), bits ? "bit-identical" : "DIFFERS");
        check(bits, "C8 history mode draws u ~ N(0, diag(tau_d) L_d L_d' diag(tau_d)) once per "
                    "new group and draw");
    }

    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "ALL PASSED", g_fail,
                g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}

// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_examples_HSGPRegression.cpp
//
//  examples/HSGPRegression.cpp: the derived keys sqrt_spd (square root of
//  the spectral density at lambda for amp = exp(log_amp), ell = exp(log_ell)),
//  f = phi (sqrt_spd z) and mu = Intercept + f, and the loader set_current.
//    C1  after construction, sqrt_spd, f and mu equal the values at the
//        initial state, and y_rep from predict_at({}) is centred on mu;
//    C2  after one sweep, set_current of Intercept, log_amp and z loads the
//        values into the joint block (its current value and the fallback
//        rows of get_history() equal the loaded values) and sqrt_spd, f and
//        mu equal the values at the loaded state;
//    C3  the next sweep starts from the loaded state: it equals the sweep of
//        a second model with the same seed and history whose block was
//        loaded with the same values through composite_block::set_current;
//    C4  a set_current call with a valid Intercept and a z of the wrong
//        length throws and leaves the state unchanged;
//    C5  set_current(get_current()) returns the same get_current() values,
//        with sqrt_spd, f and mu exact.
//    C6  keep_history = true: predict_at({}) and predict_at({phi: basis at
//        new x}) return sqrt_spd, f, mu and y_rep (the computed nodes of the
//        predict DAG) with one row per stored draw; row d of sqrt_spd, f and
//        mu is computed from Intercept, log_amp, log_ell and z of stored draw
//        d, and row d of y_rep is centred on row d of mu with sd
//        exp(log_sigma_d); predict_at(new_data, true) returns one row, equal
//        for f and mu to the last row of the history-mode output;
//    C7  keep_history = false: predict_at returns one row from the current
//        state: sqrt_spd, f, mu and y_rep, for {} at the training phi and for
//        a new phi;
//    C8  clear_history() with the predict_at outputs as per-draw buffers
//        (clear_history_contract.hpp): after k1 sweeps, clear_history() and
//        k2 sweeps, f and mu of predict_at({}) and predict_at({phi}) have k2
//        rows and equal the last k2 rows of a run without the call; y_rep
//        has k2 rows.
//  The derived values are recomputed with loops in this test (relative
//  tolerance 1e-12). Centred: the mean of (y_rep - mu) / sigma lies within
//  5 / sqrt(number of values) of 0.
// ============================================================================
#include <armadillo>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>

#include "AI4BayesCode/composite_block.hpp"

#define main hsgp_regression_demo_main
#include "../examples/HSGPRegression.cpp"
#undef main

#include "clear_history_contract.hpp"

// An explicit instantiation may name a private member: this gives the test
// access to the wrapper's composite_block.
template <typename Tag, typename Tag::type Member>
struct expose_member {
    friend typename Tag::type member_of(Tag) { return Member; }
};
struct impl_tag {
    using type = std::unique_ptr<AI4BayesCode::composite_block> HSGPRegression::*;
    friend type member_of(impl_tag);
};
template struct expose_member<impl_tag, &HSGPRegression::impl_>;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

AI4BayesCode::composite_block& impl(const HSGPRegression& m) {
    return *(m.*member_of(impl_tag{}));
}

double rel_err(const arma::vec& a, const arma::vec& b) {
    if (a.n_elem != b.n_elem) return INFINITY;
    double e = 0.0;
    for (std::size_t i = 0; i < a.n_elem; ++i)
        e = std::max(e, std::abs(a[i] - b[i]) / std::max(1.0, std::abs(b[i])));
    return e;
}

// sqrt_spd from the current log_amp, log_ell and lambda.
arma::vec spd_ref(const HSGPRegression& m) {
    const auto& d = impl(m).data();
    const double amp = std::exp(d.get("log_amp")[0]);
    const double ell = std::exp(d.get("log_ell")[0]);
    const arma::vec& lam = d.get("lambda");
    arma::vec ss(lam.n_elem);
    for (std::size_t k = 0; k < lam.n_elem; ++k)
        ss[k] = std::sqrt(amp * amp * std::sqrt(2.0 * M_PI) * ell *
                          std::exp(-0.5 * ell * ell * lam[k]));
    return ss;
}

// mu = Intercept + phi (sqrt_spd z) from the current shared values.
arma::vec mu_ref(const HSGPRegression& m) {
    const auto& d = impl(m).data();
    const arma::vec& P = d.get("phi");
    const arma::vec& z = d.get("z");
    const arma::vec ss = spd_ref(m);
    const std::size_t K = z.n_elem, N = P.n_elem / K;
    arma::vec mu(N);
    for (std::size_t n = 0; n < N; ++n) {
        double f = 0.0;
        for (std::size_t k = 0; k < K; ++k) f += ss[k] * z[k] * P[n + k * N];
        mu[n] = d.get("Intercept")[0] + f;
    }
    return mu;
}

double derived_error(const HSGPRegression& m) {
    const auto& d = impl(m).data();
    const arma::vec mu = mu_ref(m);
    return std::max(std::max(rel_err(d.get("sqrt_spd"), spd_ref(m)), rel_err(d.get("mu"), mu)),
                    rel_err(d.get("f"), mu - d.get("Intercept")[0]));
}

double yrep_z(const HSGPRegression& m) {
    const arma::mat yr = m.predict_at(AI4BayesCode::state_map{}).at("y_rep");
    const arma::vec mu = mu_ref(m);
    const double sigma = std::exp(impl(m).data().get("log_sigma")[0]);
    double s = 0.0;
    for (std::size_t n = 0; n < mu.n_elem; ++n) s += (yr[n] - mu[n]) / sigma;
    return s / std::sqrt(static_cast<double>(mu.n_elem));
}

bool same_bits(const arma::vec& a, const arma::vec& b) {
    return a.n_elem == b.n_elem &&
           std::memcmp(a.memptr(), b.memptr(), a.n_elem * sizeof(double)) == 0;
}

bool same_mat(const arma::mat& a, const arma::mat& b) {
    return a.n_rows == b.n_rows && a.n_cols == b.n_cols &&
           std::memcmp(a.memptr(), b.memptr(), a.n_elem * sizeof(double)) == 0;
}

double max_rel(const arma::mat& a, const arma::mat& b) {
    if (a.n_rows != b.n_rows || a.n_cols != b.n_cols) return INFINITY;
    double e = 0.0;
    for (std::size_t i = 0; i < a.n_elem; ++i)
        e = std::max(e, std::abs(a[i] - b[i]) / std::max(1.0, std::abs(b[i])));
    return e;
}

std::string shape(const arma::mat& a) {
    return std::to_string(a.n_rows) + "x" + std::to_string(a.n_cols);
}

// The basis at the points xs with the model's L and lambda: column m is
// sqrt(1/L) sin(sqrt(lambda_m) (xs + L)).
arma::mat basis_at(const HSGPRegression& m, const arma::vec& xs) {
    const auto& d = impl(m).data();
    const double L = d.get("L")[0];
    const arma::vec& lam = d.get("lambda");
    arma::mat P(xs.n_elem, lam.n_elem);
    for (std::size_t k = 0; k < lam.n_elem; ++k)
        for (std::size_t n = 0; n < xs.n_elem; ++n)
            P(n, k) = std::sqrt(1.0 / L) * std::sin(std::sqrt(lam[k]) * (xs[n] + L));
    return P;
}

// f and mu at the basis Pm (N_new x M), one row per row of the history h:
// row d from Intercept, log_amp, log_ell and z of row d.
// sqrt_spd_k = sqrt(amp^2 sqrt(2 pi) ell exp(-ell^2 lambda_k / 2)), one row
// per row of the history.
arma::mat spd_ref(const AI4BayesCode::history_map& h, const arma::vec& lam) {
    const arma::mat& la = h.at("log_amp");
    const arma::mat& le = h.at("log_ell");
    arma::mat ss(la.n_rows, lam.n_elem);
    for (std::size_t d = 0; d < la.n_rows; ++d) {
        const double amp = std::exp(la(d, 0)), ell = std::exp(le(d, 0));
        for (std::size_t k = 0; k < lam.n_elem; ++k)
            ss(d, k) = std::sqrt(amp * amp * std::sqrt(2.0 * M_PI) * ell *
                                 std::exp(-0.5 * ell * ell * lam[k]));
    }
    return ss;
}

void f_mu_ref(const AI4BayesCode::history_map& h, const arma::vec& lam,
              const arma::mat& Pm, arma::mat& f, arma::mat& mu) {
    const arma::mat& I  = h.at("Intercept");
    const arma::mat& la = h.at("log_amp");
    const arma::mat& le = h.at("log_ell");
    const arma::mat& z  = h.at("z");
    f.set_size(I.n_rows, Pm.n_rows);
    mu.set_size(I.n_rows, Pm.n_rows);
    for (std::size_t d = 0; d < I.n_rows; ++d) {
        const double amp = std::exp(la(d, 0)), ell = std::exp(le(d, 0));
        arma::vec ss(lam.n_elem);
        for (std::size_t k = 0; k < lam.n_elem; ++k)
            ss[k] = std::sqrt(amp * amp * std::sqrt(2.0 * M_PI) * ell *
                              std::exp(-0.5 * ell * ell * lam[k]));
        for (std::size_t n = 0; n < Pm.n_rows; ++n) {
            double v = 0.0;
            for (std::size_t k = 0; k < Pm.n_cols; ++k) v += ss[k] * z(d, k) * Pm(n, k);
            f(d, n)  = v;
            mu(d, n) = I(d, 0) + v;
        }
    }
}

// Sum over every entry of (y_rep - mu) / exp(log_sigma_d), divided by
// sqrt(number of entries).
double yrep_z_rows(const arma::mat& yr, const arma::mat& mu, const arma::mat& log_sigma) {
    double s = 0.0;
    for (std::size_t d = 0; d < yr.n_rows; ++d)
        for (std::size_t n = 0; n < yr.n_cols; ++n)
            s += (yr(d, n) - mu(d, n)) / std::exp(log_sigma(d, 0));
    return s / std::sqrt(static_cast<double>(yr.n_elem));
}

// C8: a keep_history model whose per-draw buffers are get_history() and the
// predict_at outputs at the training and at the new basis.
clear_history_check::model contract_model(const arma::vec& y, const arma::vec& x,
                                          int K, const AI4BayesCode::state_map& nd_new) {
    auto w = std::make_shared<HSGPRegression>(y, x, K, 13, true);
    HSGPRegression* p = w.get();
    clear_history_check::model c;
    c.step    = [p](int n) { p->step(n); };
    c.clear   = [p] { p->clear_history(); };
    c.history = [p](clear_history_check::buffers& b) {
        for (const auto& kv : p->get_history()) b.mats[kv.first] = kv.second;
    };
    c.extra   = [p, nd_new](clear_history_check::buffers& b) {
        const AI4BayesCode::state_map none;
        for (const AI4BayesCode::state_map* in : {&none, &nd_new}) {
            const std::string tag = in->empty() ? "predict_at({}) " : "predict_at({phi}) ";
            for (const auto& kv : p->predict_at(*in)) {
                if (kv.first == "y_rep") b.rows_only[tag + kv.first] = kv.second;
                else                     b.mats[tag + kv.first] = kv.second;
            }
        }
    };
    c.sizes   = [p] {
        AI4BayesCode::composite_block& cb = impl(*p);
        std::vector<std::size_t> s{cb.history_size()};
        for (std::size_t i = 0; i < cb.n_children(); ++i) s.push_back(cb.child(i).history_size());
        return s;
    };
    c.owners.push_back(w);
    return c;
}

}  // namespace

int main() {
    std::printf("=== test_fix_examples_HSGPRegression ===\n");
    const std::size_t N = 150, K = 6;
    std::mt19937_64 gen(29u);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::uniform_real_distribution<double> ud(-1.0, 1.0);
    arma::vec x(N), y(N);
    for (std::size_t n = 0; n < N; ++n) {
        x[n] = ud(gen);
        y[n] = 3.0 + std::sin(3.0 * x[n]) + 0.2 * nd(gen);
    }

    HSGPRegression m(y, x, static_cast<int>(K), 7, false);
    std::printf("C1 construction: max rel error of sqrt_spd, f, mu = %.3g, y_rep z = %.3f\n",
                derived_error(m), yrep_z(m));
    check(derived_error(m) <= 1e-12, "C1 sqrt_spd, f and mu equal the values at the initial state");
    check(std::abs(yrep_z(m)) < 5.0, "C1 predict_at({}) y_rep centred on mu");

    m.step(1);
    AI4BayesCode::state_map s;
    s["Intercept"] = arma::vec{-4.0};
    s["log_amp"]   = arma::vec{0.3};
    s["z"]         = arma::vec{1.0, -1.0, 0.5, 2.0, -0.5, 0.25};
    m.set_current(s);
    const arma::vec cur = impl(m).child(0).current();
    const auto hist = m.get_history();
    const bool loaded = cur[0] == -4.0 && cur[1] == 0.3 &&
                        same_bits(cur.subvec(4, 4 + K - 1), s.at("z")) &&
                        hist.at("Intercept")(0, 0) == -4.0 &&
                        hist.at("log_amp")(0, 0) == 0.3;
    std::printf("C2 set_current: block Intercept %.3f log_amp %.3f, history row "
                "Intercept %.3f; max rel error of sqrt_spd, f, mu = %.3g\n",
                cur[0], cur[1], hist.at("Intercept")(0, 0), derived_error(m));
    check(loaded, "C2 the joint block and get_history() hold the loaded values");
    check(derived_error(m) <= 1e-12, "C2 sqrt_spd, f and mu equal the values at the loaded state");

    // C3: a twin with the same seed and history, loaded through
    // composite_block::set_current (the per-child loader of the framework).
    HSGPRegression t(y, x, static_cast<int>(K), 7, false);
    t.step(1);
    {
        AI4BayesCode::composite_block& c = impl(t);
        arma::vec theta = c.child(0).current();
        theta[0] = -4.0; theta[1] = 0.3;
        theta.subvec(4, 4 + K - 1) = s.at("z");
        c.set_current(theta);
    }
    m.step(1); t.step(1);
    const auto cm = m.get_current(), ct = t.get_current();
    bool same = true;
    for (const char* k : {"Intercept", "log_amp", "log_ell", "log_sigma", "z"})
        same = same && same_bits(cm.at(k), ct.at(k));
    check(same, "C3 the next sweep starts from the loaded state");

    // C4
    const auto before = m.get_current();
    const arma::vec mu0 = impl(m).data().get("mu");
    AI4BayesCode::state_map bad;
    bad["Intercept"] = arma::vec{9.0};
    bad["z"]         = arma::vec{1.0, 2.0};
    bool threw = false;
    try { m.set_current(bad); } catch (const std::exception&) { threw = true; }
    const auto after = m.get_current();
    const bool unchanged = same_bits(before.at("Intercept"), after.at("Intercept")) &&
                           same_bits(before.at("z"), after.at("z")) &&
                           impl(m).child(0).current()[0] == before.at("Intercept")[0] &&
                           same_bits(mu0, impl(m).data().get("mu"));
    std::printf("C4 failing set_current: threw = %d, state unchanged = %d\n",
                threw ? 1 : 0, unchanged ? 1 : 0);
    check(threw && unchanged, "C4 a failing set_current leaves the state unchanged");

    // C5
    m.step(3);
    const auto b5 = m.get_current();
    m.set_current(b5);
    const auto a5 = m.get_current();
    bool same5 = true;
    for (const char* k : {"Intercept", "log_amp", "log_ell", "log_sigma", "z", "amp", "ell", "sigma"})
        same5 = same5 && same_bits(b5.at(k), a5.at(k));
    check(same5 && derived_error(m) <= 1e-12,
          "C5 set_current(get_current()) leaves the state unchanged and sqrt_spd, f, mu exact");

    // C6: history mode.
    const std::size_t N_new = 9;
    arma::vec x_new(N_new);
    for (std::size_t n = 0; n < N_new; ++n) x_new[n] = ud(gen);
    const arma::mat Pn = basis_at(m, x_new);
    const arma::mat P_tr = basis_at(m, x);
    const arma::vec lam = impl(m).data().get("lambda");
    AI4BayesCode::state_map nd_new;
    nd_new["phi"] = arma::vectorise(Pn);
    {
        const std::size_t D = 20;
        HSGPRegression h6(y, x, static_cast<int>(K), 11, true);
        h6.step(static_cast<int>(D));
        const auto hist = h6.get_history();
        const auto p_tr = h6.predict_at(AI4BayesCode::state_map{});
        const auto p_nw = h6.predict_at(nd_new);
        bool keys = p_tr.size() == 4 && p_nw.size() == 4;
        for (const char* k : {"sqrt_spd", "f", "mu", "y_rep"})
            keys = keys && p_tr.count(k) && p_nw.count(k);
        check(keys, "C6 history mode: predict_at returns sqrt_spd, f, mu and y_rep");
        if (keys) {
            bool rows = p_tr.at("sqrt_spd").n_rows == D && p_tr.at("sqrt_spd").n_cols == K &&
                        p_nw.at("sqrt_spd").n_rows == D && p_nw.at("sqrt_spd").n_cols == K;
            for (const char* k : {"f", "mu", "y_rep"})
                rows = rows && p_tr.at(k).n_rows == D && p_tr.at(k).n_cols == N &&
                       p_nw.at(k).n_rows == D && p_nw.at(k).n_cols == N_new;
            arma::mat f_tr, mu_tr, f_nw, mu_nw;
            f_mu_ref(hist, lam, P_tr, f_tr, mu_tr);
            f_mu_ref(hist, lam, Pn, f_nw, mu_nw);
            const arma::mat ss = spd_ref(hist, lam);
            const double e = std::max(std::max(
                std::max(max_rel(p_tr.at("f"), f_tr), max_rel(p_tr.at("mu"), mu_tr)),
                std::max(max_rel(p_nw.at("f"), f_nw), max_rel(p_nw.at("mu"), mu_nw))),
                std::max(max_rel(p_tr.at("sqrt_spd"), ss), max_rel(p_nw.at("sqrt_spd"), ss)));
            const double z_tr = yrep_z_rows(p_tr.at("y_rep"), p_tr.at("mu"), hist.at("log_sigma"));
            const double z_nw = yrep_z_rows(p_nw.at("y_rep"), p_nw.at("mu"), hist.at("log_sigma"));
            std::printf("C6 history mode: y_rep %s ({}) and %s ({phi}); max rel error of "
                        "f, mu = %.3g; y_rep z = %.3f, %.3f\n",
                        shape(p_tr.at("y_rep")).c_str(), shape(p_nw.at("y_rep")).c_str(),
                        e, z_tr, z_nw);
            check(rows, "C6 one row per stored draw: D x N for {}, D x N_new for {phi}");
            check(e <= 1e-12, "C6 row d of sqrt_spd, f and mu from Intercept, log_amp, "
                              "log_ell and z of stored draw d");
            check(std::abs(z_tr) < 5.0 && std::abs(z_nw) < 5.0,
                  "C6 row d of y_rep centred on row d of mu, sd exp(log_sigma_d)");
            const auto p_last = h6.predict_at(nd_new, true);
            const bool last_ok =
                p_last.size() == 4 && p_last.at("y_rep").n_rows == 1 &&
                p_last.at("y_rep").n_cols == N_new &&
                same_mat(p_last.at("f"), arma::mat(p_nw.at("f").row(D - 1))) &&
                same_mat(p_last.at("mu"), arma::mat(p_nw.at("mu").row(D - 1)));
            check(last_ok, "C6 last_draw_only = true: one row; f, mu = last row of the "
                           "history-mode output");
        }
    }

    // C7: keep_history = false (model m): one row from the current state.
    {
        const auto p0 = m.predict_at(AI4BayesCode::state_map{});
        const auto p1 = m.predict_at(nd_new);
        arma::mat f_ref, mu_ref1, f_ref0, mu_ref0;
        f_mu_ref(m.get_history(), lam, Pn, f_ref, mu_ref1);   // 1-row history = current state
        f_mu_ref(m.get_history(), lam, P_tr, f_ref0, mu_ref0);
        const arma::mat ss_cur = spd_ref(m.get_history(), lam);
        const bool ok = p0.size() == 4 && p0.count("y_rep") &&
                        p0.at("y_rep").n_rows == 1 && p0.at("y_rep").n_cols == N &&
                        max_rel(p0.at("f"), f_ref0) <= 1e-12 &&
                        max_rel(p0.at("mu"), mu_ref0) <= 1e-12 &&
                        max_rel(p0.at("sqrt_spd"), ss_cur) <= 1e-12 &&
                        p1.size() == 4 && p1.at("y_rep").n_rows == 1 &&
                        p1.at("y_rep").n_cols == N_new &&
                        max_rel(p1.at("f"), f_ref) <= 1e-12 &&
                        max_rel(p1.at("mu"), mu_ref1) <= 1e-12 &&
                        max_rel(p1.at("sqrt_spd"), ss_cur) <= 1e-12;
        std::printf("C7 keep_history = false: {} -> y_rep %s; {phi} -> f %s, mu %s, y_rep %s\n",
                    shape(p0.at("y_rep")).c_str(), shape(p1.at("f")).c_str(),
                    shape(p1.at("mu")).c_str(), shape(p1.at("y_rep")).c_str());
        check(ok, "C7 keep_history = false: 1 x N rows from the current state");
    }

    // C8: clear_history() with the predict_at outputs.
    clear_history_check::run("C8 clear_history() and predict_at",
                             [&] { return contract_model(y, x, static_cast<int>(K), nd_new); },
                             40, 30);

    const int n_fail = g_fail + clear_history_check::failures();
    std::printf("%s (%d failure%s)\n", n_fail ? "FAILED" : "ALL PASSED", n_fail,
                n_fail == 1 ? "" : "s");
    return n_fail ? 1 : 0;
}

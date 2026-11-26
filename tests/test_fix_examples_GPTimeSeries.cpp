// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_examples_GPTimeSeries.cpp
//
//  examples/GPTimeSeries.cpp: celerite_logp (log p(y | amp, tau, sigma),
//  reported by get_current() as "logp"), predict_at at new times and at
//  the training times, and the predict DAG of get_dag().
//  Reference values are computed densely in this test: K = a exp(-c |dt|)
//  + (sigma^2 + 1e-10) I with a = amp^2, c = 1 / tau. Data: (t, y) with
//  N = 60, t_i = 0.25 i, for C1-C6, C9-C12 and C15; (t40, y40) with N = 40,
//  t_i = i, for C7, C8, C13, C14 and C16, where every model is started at
//  (amp, tau, sigma) = (1.5, 4, 0.5) by set_current so that the posterior of
//  f has correlated neighbours.
//    C1  after construction, logp equals log N(y; 0, K) at the initial
//        amp, tau, sigma (relative tolerance 1e-8);
//    C2  after one sweep and set_current of amp, tau and sigma, logp equals
//        the value at the loaded hyperparameters;
//    C3  set_current with a valid amp and a non-positive tau throws and
//        leaves amp, a_real and logp unchanged;
//    C4  with keep_history, predict_at({t: new times}) in either mode
//        leaves get_history() unchanged;
//    C5  in history mode, f_mean of draw d at the new times equals the GP
//        posterior mean K(t*, t) K^-1 y at (amp_d, tau_d, sigma_d)
//        (relative tolerance 1e-6);
//    C6  set_current with an empty amp, tau or sigma throws and leaves
//        get_current() unchanged;
//    C7  in history mode, predict_at({}) returns f_mean, f_sd, f_star and
//        y_rep with one row per stored draw at the training t. For draw d,
//        f_mean_i and f_sd_i^2 equal the dense posterior mean m_i and
//        variance C_ii of f(t_i) given y at (amp_d, tau_d, sigma_d)
//        (relative tolerance 1e-9, absolute 1e-9 * amp_d^2), and f_star and
//        y_rep equal the dense replay of a joint draw (replay_row): f at the
//        nodes in decreasing time order is m + L z, L the lower Cholesky
//        factor of C in that order, z from an engine seeded as predict_rng_,
//        then y_rep_i = f_star_i + sigma_d * e_i (absolute tolerance
//        1e-8 * amp_d);
//    C8  predict_at({}, last_draw_only = true) gives the same at the
//        current state in one row;
//    C9  the y_rep stochastic refresher, run by composite_block::predict_at
//        from an engine seeded as predict_rng_, equals the y_rep of the
//        single-draw predict_at bit for bit, at the training t and at new
//        times;
//    C10 get_dag() declares the one data input "t", and predict_at with any
//        other key throws an error that names the key;
//    C11 get_dag() predict edges are amp -> a_real, tau -> c_real,
//        sigma -> obs_diag and t, a_real, c_real, obs_diag, sigma -> y_rep
//        (the keys passed to d.get in the refreshers), and the context edges
//        are the three prior edges;
//    C12 composite_block::predict_at with a replaced amp recomputes
//        a_real = amp^2 and draws y_rep from it: y_rep equals, bit for bit,
//        the single-draw predict_at of a twin model after set_current(amp);
//    C13 at new times that are unsorted, repeated, equal to training times
//        and outside the training range, history-mode and single-draw
//        predict_at give f_mean, f_sd, f_star and y_rep equal to the dense
//        posterior and the dense replay over the nodes from the largest to
//        the smallest new time (the training times in between included);
//        repeated times give identical f_star; f_mean and f_sd also equal
//        celerite_gp_block::predict_mean_var (relative tolerance 1e-9);
//    C14 N = 40, at the training t: over 20000 single-draw calls at a fixed
//        (amp, tau, sigma), and over 10000 history-mode calls for each
//        stored draw d, the sample mean and covariance of f_star match m_d
//        and C_d, and those of y_rep match m_d and C_d + sigma_d^2 I (every
//        entry within 6 standard errors);
//    C15 predict_at with a non-finite time throws an error that names the
//        entry, in both modes;
//    C16 with tied training times, predict_at({}) matches the dense
//        posterior and replay, and tied times give identical f_star;
//    C17 at amp in [1e-150, 1e150], tau in [1e-9, 1e12] and sigma in
//        [1e-12, 1e6] (200 single-draw calls, training and new times),
//        every output is finite and f_sd >= 0.
// ============================================================================
#include <armadillo>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "AI4BayesCode/composite_block.hpp"

#define main gp_time_series_demo_main
#include "../examples/GPTimeSeries.cpp"
#undef main

// An explicit instantiation may name a private member: this gives the test
// read access to the wrapper's composite_block.
template <typename Tag, typename Tag::type Member>
struct expose_member {
    friend typename Tag::type member_of(Tag) { return Member; }
};
struct impl_tag {
    using type = std::unique_ptr<AI4BayesCode::composite_block> GPTimeSeries::*;
    friend type member_of(impl_tag);
};
template struct expose_member<impl_tag, &GPTimeSeries::impl_>;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

bool same_bits(const arma::mat& a, const arma::mat& b) {
    return a.n_elem == b.n_elem &&
           std::memcmp(a.memptr(), b.memptr(), a.n_elem * sizeof(double)) == 0;
}

arma::mat cov(const arma::vec& s, const arma::vec& t, double amp, double tau) {
    arma::mat K(s.n_elem, t.n_elem);
    for (std::size_t i = 0; i < s.n_elem; ++i)
        for (std::size_t j = 0; j < t.n_elem; ++j)
            K(i, j) = amp * amp * std::exp(-std::abs(s[i] - t[j]) / tau);
    return K;
}

double dense_logp(const arma::vec& t, const arma::vec& y, double amp, double tau, double sigma) {
    arma::mat K = cov(t, t, amp, tau);
    K.diag() += sigma * sigma + 1e-10;
    const arma::mat L = arma::chol(K, "lower");
    const arma::vec a = arma::solve(arma::trimatl(L), y);
    return -0.5 * (y.n_elem * std::log(2.0 * M_PI) + 2.0 * arma::sum(arma::log(L.diag())) +
                   arma::dot(a, a));
}

double rel(double a, double b) { return std::abs(a - b) / std::max(1.0, std::abs(b)); }

// Dense posterior of f at the times s given the training (t, y):
// mean K(s, t) Ky^-1 y and covariance K(s, s) - K(s, t) Ky^-1 K(t, s), with
// Ky = K(t, t) + (sigma^2 + 1e-10) I.
struct dense_post_t {
    arma::vec m;
    arma::mat C;
};
dense_post_t dense_post(const arma::vec& s, const arma::vec& t, const arma::vec& y,
                        double amp, double tau, double sigma) {
    arma::mat Ky = cov(t, t, amp, tau);
    Ky.diag() += sigma * sigma + 1e-10;
    const arma::mat Kst = cov(s, t, amp, tau);
    dense_post_t p;
    p.m = Kst * arma::solve(Ky, y);
    p.C = cov(s, s, amp, tau) - Kst * arma::solve(Ky, Kst.t());
    p.C = 0.5 * (p.C + p.C.t());
    return p;
}

// Dense replay of one predict_at row at the times tq. The nodes are the
// distinct values of tq and of the training times in [min tq, max tq], in
// decreasing order; f at the nodes is m + L z with L the lower Cholesky
// factor of C in that order and z_1, z_2, ... from (r, z01); then
// y_rep_i = f_star_i + sigma * e_i with e_1, e_2, ... from (r, z01) in the
// order of tq. Returns false if C is not numerically positive definite.
bool replay_row(const arma::vec& tq, const arma::vec& t, const arma::vec& y,
                double amp, double tau, double sigma, std::mt19937_64& r,
                std::normal_distribution<double>& z01, arma::vec& f_star,
                arma::vec& y_rep) {
    const double lo = tq.min(), hi = tq.max();
    std::set<double> S(tq.begin(), tq.end());
    for (double tj : t)
        if (tj >= lo && tj <= hi) S.insert(tj);
    arma::vec nodes(S.size());
    std::size_t k = S.size();
    for (double s : S) nodes[--k] = s;
    const dense_post_t p = dense_post(nodes, t, y, amp, tau, sigma);
    arma::mat L;
    if (!arma::chol(L, p.C, "lower")) return false;
    arma::vec z(nodes.n_elem);
    for (std::size_t i = 0; i < z.n_elem; ++i) z[i] = z01(r);
    const arma::vec x = p.m + L * z;
    f_star.set_size(tq.n_elem);
    y_rep.set_size(tq.n_elem);
    for (std::size_t i = 0; i < tq.n_elem; ++i) {
        std::size_t kk = 0;
        while (nodes[kk] != tq[i]) ++kk;
        f_star[i] = x[kk];
    }
    for (std::size_t i = 0; i < tq.n_elem; ++i) y_rep[i] = f_star[i] + sigma * z01(r);
    return true;
}

// Compares the rows of p (times tq) with the dense posterior and the dense
// replay, rows d = 0, 1, ... with hyperparameters (A, T, S)(d, 0) and one
// replay engine seeded as predict_rng_ of the model with rng_seed `seed`.
// Returns true when f_mean and f_sd^2 are within relative 1e-9 (absolute
// 1e-9 * amp^2 for the variance) and f_star and y_rep within 1e-8 * amp.
bool check_rows(const AI4BayesCode::history_map& p, const arma::vec& tq, const arma::vec& t,
                const arma::vec& y, const arma::mat& A, const arma::mat& T, const arma::mat& S,
                std::uint64_t seed, const char* label) {
    const std::size_t nr = A.n_rows, n = tq.n_elem;
    bool shape = p.size() == 7;
    for (const char* k : {"f_mean", "f_sd", "f_star", "y_rep"})
        shape = shape && p.count(k) && p.at(k).n_rows == nr && p.at(k).n_cols == n;
    for (const char* k : {"a_real", "c_real", "obs_diag"})
        shape = shape && p.count(k) && p.at(k).n_rows == nr && p.at(k).n_cols == 1;
    if (!shape) {
        std::printf("%s: wrong keys or shapes\n", label);
        return false;
    }
    const arma::mat& M = p.at("f_mean");
    const arma::mat& SD = p.at("f_sd");
    const arma::mat& FS = p.at("f_star");
    const arma::mat& YR = p.at("y_rep");
    double worst_m = 0.0, worst_v = 0.0, worst_f = 0.0, worst_y = 0.0;
    bool chol_ok = true;
    std::mt19937_64 r(seed ^ 0x9E3779B97F4A7C15ULL);
    std::normal_distribution<double> z01(0.0, 1.0);
    for (std::size_t d = 0; d < nr; ++d) {
        const double amp = A(d, 0), tau = T(d, 0), sig = S(d, 0);
        const dense_post_t q = dense_post(tq, t, y, amp, tau, sig);
        arma::vec fs, yr;
        chol_ok = chol_ok && replay_row(tq, t, y, amp, tau, sig, r, z01, fs, yr);
        if (!chol_ok) break;
        for (std::size_t i = 0; i < n; ++i) {
            worst_m = std::max(worst_m, rel(M(d, i), q.m[i]));
            worst_v = std::max(worst_v, std::abs(SD(d, i) * SD(d, i) - q.C(i, i)) /
                                            std::max(amp * amp, std::abs(q.C(i, i))));
            worst_f = std::max(worst_f, std::abs(FS(d, i) - fs[i]) / amp);
            worst_y = std::max(worst_y, std::abs(YR(d, i) - yr[i]) / amp);
        }
    }
    std::printf("%s: %zu row(s) x %zu; max rel |f_mean - m| = %.3g, "
                "max |f_sd^2 - C_ii| / amp^2 = %.3g, replay: max |f_star - f| / amp = %.3g, "
                "max |y_rep - y| / amp = %.3g%s\n",
                label, nr, n, worst_m, worst_v, worst_f, worst_y,
                chol_ok ? "" : " (Cholesky of C FAILED)");
    return chol_ok && worst_m <= 1e-9 && worst_v <= 1e-9 && worst_f <= 1e-8 && worst_y <= 1e-8;
}

}  // namespace

int main() {
    // Unbuffered, so the lines printed before a crash reach the log.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("=== test_fix_examples_GPTimeSeries ===\n");
    const std::size_t N = 60;
    std::mt19937_64 g(71u);
    std::normal_distribution<double> nd(0.0, 1.0);
    arma::vec t(N), y(N);
    for (std::size_t i = 0; i < N; ++i) {
        t[i] = 0.25 * static_cast<double>(i);
        y[i] = std::sin(0.8 * t[i]) + 0.2 * nd(g);
    }

    GPTimeSeries m(t, y, 7, false);
    auto& c = *(m.*member_of(impl_tag{}));
    auto ref_now = [&](const GPTimeSeries& mm) {
        const auto cur = mm.get_current();
        return dense_logp(t, y, cur.at("amp")[0], cur.at("tau")[0], cur.at("sigma")[0]);
    };
    std::printf("C1 construction: logp %.6f, dense %.6f\n", m.get_current().at("logp")[0], ref_now(m));
    check(rel(m.get_current().at("logp")[0], ref_now(m)) <= 1e-8,
          "C1 logp is the log marginal likelihood at the initial state");

    m.step(1);
    AI4BayesCode::state_map s;
    s["amp"] = arma::vec{2.0}; s["tau"] = arma::vec{0.5}; s["sigma"] = arma::vec{0.4};
    m.set_current(s);
    std::printf("C2 set_current: logp %.6f, dense %.6f\n", m.get_current().at("logp")[0], ref_now(m));
    check(rel(m.get_current().at("logp")[0], ref_now(m)) <= 1e-8,
          "C2 logp is the log marginal likelihood at the loaded state");

    const arma::vec amp0 = c.data().get("amp"), ar0 = c.data().get("a_real"),
                    lp0 = c.data().get("celerite_logp");
    AI4BayesCode::state_map bad;
    bad["amp"] = arma::vec{5.0}; bad["tau"] = arma::vec{-1.0};
    bool threw = false;
    try { m.set_current(bad); } catch (const std::exception&) { threw = true; }
    check(threw && same_bits(amp0, c.data().get("amp")) && same_bits(ar0, c.data().get("a_real")) &&
          same_bits(lp0, c.data().get("celerite_logp")),
          "C3 a failing set_current leaves amp, a_real and logp unchanged");

    // C4, C5
    GPTimeSeries h(t, y, 9, true);
    h.step(25);
    const auto hist0 = h.get_history();
    AI4BayesCode::state_map at;
    at["t"] = arma::vec{1.1, 3.3, 7.7, 12.0};
    const auto pred = h.predict_at(at);                 // history mode
    (void)h.predict_at(at, /*last_draw_only=*/true);    // stateful mode
    const auto hist1 = h.get_history();
    bool hist_same = hist0.size() == hist1.size();
    for (const auto& kv : hist0)
        hist_same = hist_same && hist1.count(kv.first) && same_bits(kv.second, hist1.at(kv.first));
    std::printf("C4 history rows: celerite_logp %u before, %u after predict_at\n",
                static_cast<unsigned>(hist0.at("celerite_logp").n_rows),
                static_cast<unsigned>(hist1.at("celerite_logp").n_rows));
    check(hist_same, "C4 predict_at leaves get_history() unchanged");

    const arma::mat& A = hist0.at("amp");
    const arma::mat& T = hist0.at("tau");
    const arma::mat& S = hist0.at("sigma");
    const arma::mat& F = pred.at("f_mean");
    double worst = 0.0;
    for (std::size_t d = 0; d < A.n_rows; ++d) {
        arma::mat K = cov(t, t, A(d, 0), T(d, 0));
        K.diag() += S(d, 0) * S(d, 0) + 1e-10;
        const arma::vec mu = cov(at.at("t"), t, A(d, 0), T(d, 0)) * arma::solve(K, y);
        for (std::size_t i = 0; i < mu.n_elem; ++i) worst = std::max(worst, rel(F(d, i), mu[i]));
    }
    std::printf("C5 history predict_at: max rel |f_mean - dense mean of draw d| = %.3g\n", worst);
    check(worst <= 1e-6, "C5 history-mode f_mean uses the hyperparameters of each draw");

    {
        GPTimeSeries me(t, y, 17, false);
        me.step(2);
        bool ok = true;
        for (const char* key : {"amp", "tau", "sigma"}) {
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
        check(ok, "C6 set_current with an empty amp, tau or sigma throws and leaves get_current() unchanged");
    }

    // Data of C7, C8, C13, C14 and C16: N = 40, t_i = i. Each model there is
    // started at (amp, tau, sigma) = start40 by set_current, so that the
    // posterior of f has correlated neighbours; from the default start, the
    // chains on (t, y) above reach sigma < 1e-3 within a few sweeps, where
    // C is close to sigma^2 I.
    const std::size_t N40 = 40;
    arma::vec t40(N40), y40(N40);
    {
        std::mt19937_64 g40(79u);
        std::normal_distribution<double> nd40(0.0, 1.0);
        for (std::size_t i = 0; i < N40; ++i) {
            t40[i] = static_cast<double>(i);
            y40[i] = 1.5 * std::sin(0.2 * t40[i]) + 0.5 * nd40(g40);
        }
    }
    AI4BayesCode::state_map start40;
    start40["amp"] = arma::vec{1.5};
    start40["tau"] = arma::vec{4.0};
    start40["sigma"] = arma::vec{0.5};
    auto one_row = [](double v) { return arma::mat(1, 1, arma::fill::value(v)); };
    auto print_draws = [](const char* label, const AI4BayesCode::history_map& H) {
        std::printf("%s (amp, tau, sigma):", label);
        for (std::size_t d = 0; d < H.at("amp").n_rows; ++d)
            std::printf(" (%.3f, %.3f, %.3f)", H.at("amp")(d, 0), H.at("tau")(d, 0),
                        H.at("sigma")(d, 0));
        std::printf("\n");
    };

    // C7, C8: the posterior predictive at the training t.
    const std::uint64_t pred_xor = 0x9E3779B97F4A7C15ULL;   // predict_rng_ seed
    {
        GPTimeSeries h7(t40, y40, 45, true);
        h7.set_current(start40);
        h7.step(3);
        const auto hist7 = h7.get_history();
        print_draws("C7 stored draws", hist7);
        const auto p7 = h7.predict_at(AI4BayesCode::state_map{});
        GPTimeSeries h8(t40, y40, 15, true);
        h8.set_current(start40);
        const auto p8 = h8.predict_at(AI4BayesCode::state_map{}, /*last_draw_only=*/true);

        check(check_rows(p7, t40, t40, y40, hist7.at("amp"), hist7.at("tau"), hist7.at("sigma"),
                         45, "C7 history"),
              "C7 history-mode predict_at({}) at the training t: f_mean and f_sd are the marginal "
              "posterior mean and sd, f_star is one joint draw from N(m_d, C_d), "
              "y_rep = f_star + sigma_d * e");
        check(check_rows(p8, t40, t40, y40, one_row(1.5), one_row(4.0), one_row(0.5), 15,
                         "C8 single draw"),
              "C8 single-draw predict_at({}) gives the same at the current state");
    }

    // C9: the y_rep refresher of the composite equals the single-draw path.
    {
        GPTimeSeries wa(t, y, 17, false), wb(t, y, 17, false);
        wa.step(5);
        wb.step(5);
        auto& cb = *(wb.*member_of(impl_tag{}));
        std::mt19937_64 r9(static_cast<std::uint64_t>(17) ^ pred_xor);
        const arma::mat w_tr = wa.predict_at(AI4BayesCode::state_map{}).at("y_rep");
        const arma::vec c_tr = cb.predict_at(AI4BayesCode::block_context{}, r9).at("y_rep");
        AI4BayesCode::state_map tn;
        tn["t"] = arma::vec{1.1, 3.3, 7.7, 12.0, 20.0};
        const arma::mat w_new = wa.predict_at(tn).at("y_rep");
        const arma::vec c_new = cb.predict_at(tn, r9).at("y_rep");
        const bool ok_tr = w_tr.n_rows == 1 && same_bits(w_tr.row(0).t(), c_tr);
        const bool ok_new = w_new.n_rows == 1 && same_bits(w_new.row(0).t(), c_new);
        std::printf("C9 refresher vs single-draw predict_at: training t %s, new t %s\n",
                    ok_tr ? "bit-identical" : "DIFFERS", ok_new ? "bit-identical" : "DIFFERS");
        check(ok_tr && ok_new,
              "C9 the y_rep refresher equals the single-draw predict_at at the training and new t");
    }

    // C10: the declared data input and the unknown-key error.
    {
        GPTimeSeries m10(t, y, 19, true);
        m10.step(3);
        const auto dag = m10.get_dag();
        bool named = true;
        for (const char* key : {"t_new_flag", "y"}) {
            AI4BayesCode::state_map bad;
            bad[key] = arma::vec{1.0};
            std::string msg;
            try { (void)m10.predict_at(bad); } catch (const std::exception& e) { msg = e.what(); }
            std::printf("C10 predict_at({%s}): %s\n", key, msg.empty() ? "(no error)" : msg.c_str());
            named = named && msg.find(std::string("'") + key + "'") != std::string::npos;
        }
        check(dag.data_inputs == std::vector<std::string>{"t"} && named,
              "C10 data_inputs is {t}; predict_at with another key throws naming the key");
    }

    // C11: the sources of the predict edges into each node are the keys
    // passed to d.get in that node's refresher.
    {
        GPTimeSeries m11(t, y, 21, false);
        const auto dag = m11.get_dag();
        using edge_set = std::set<std::pair<std::string, std::string>>;
        const edge_set want_pe = {{"amp", "a_real"},   {"tau", "c_real"},
                                  {"sigma", "obs_diag"}, {"sigma", "y_rep"},
                                  {"t", "y_rep"},        {"a_real", "y_rep"},
                                  {"c_real", "y_rep"},   {"obs_diag", "y_rep"}};
        const edge_set want_ce = {{"amp_prior_sd", "amp"}, {"tau_prior_shape", "tau"},
                                  {"tau_prior_scale", "tau"}};
        edge_set pe, ce;
        for (const auto& kv : dag.predict_edges)
            for (const auto& to : kv.second) pe.insert({kv.first, to});
        for (const auto& kv : dag.context_edges)
            for (const auto& to : kv.second) ce.insert({kv.first, to});
        std::printf("C11 get_dag: %zu predict edges, %zu context edges\n", pe.size(), ce.size());
        check(pe == want_pe && ce == want_ce,
              "C11 predict edges amp->a_real, tau->c_real, sigma->obs_diag, "
              "{t, a_real, c_real, obs_diag, sigma}->y_rep; context edges are the priors");
    }

    // C12: composite_block::predict_at with a replaced amp recomputes a_real
    // and y_rep.
    {
        GPTimeSeries wa(t, y, 23, false), wb(t, y, 23, false);
        wa.step(4);
        wb.step(4);
        auto& ca = *(wa.*member_of(impl_tag{}));
        const double amp_new = 1.7;
        AI4BayesCode::block_context rep;
        rep["amp"] = arma::vec{amp_new};
        std::mt19937_64 r12(static_cast<std::uint64_t>(23) ^ pred_xor);
        const auto out = ca.predict_at(rep, r12);
        AI4BayesCode::state_map s12;
        s12["amp"] = arma::vec{amp_new};
        wb.set_current(s12);
        const arma::mat yb = wb.predict_at(AI4BayesCode::state_map{}, true).at("y_rep");
        const bool ar_ok = out.count("a_real") && out.at("a_real").n_elem == 1 &&
                           out.at("a_real")[0] == amp_new * amp_new;
        const bool yr_ok = out.count("y_rep") && yb.n_rows == 1 &&
                           same_bits(yb.row(0).t(), out.at("y_rep"));
        std::printf("C12 predict_at({amp}): a_real %s, y_rep %s the twin's single-draw y_rep\n",
                    ar_ok ? "= amp^2" : "WRONG", yr_ok ? "bit-identical to" : "DIFFERS from");
        check(ar_ok && yr_ok,
              "C12 composite predict_at with a replaced amp recomputes a_real; its y_rep "
              "equals the twin's after set_current");
    }

    // C13: new times, unsorted, repeated, equal to training times and
    // outside the training range [0, 39].
    {
        const arma::vec tq{17.3, -2.0, 3.5, 17.3, 45.5, 0.0, 39.0, 5.125, 3.5, 21.0};
        GPTimeSeries h13(t40, y40, 25, true);
        h13.set_current(start40);
        h13.step(3);
        const auto hist13 = h13.get_history();
        print_draws("C13 stored draws", hist13);
        AI4BayesCode::state_map nq;
        nq["t"] = tq;
        const auto p13 = h13.predict_at(nq);
        GPTimeSeries s13(t40, y40, 27, false);
        s13.set_current(start40);
        const auto q13 = s13.predict_at(nq, /*last_draw_only=*/true);
        const bool ok_h = check_rows(p13, tq, t40, y40, hist13.at("amp"), hist13.at("tau"),
                                     hist13.at("sigma"), 25, "C13 history");
        const bool ok_s = check_rows(q13, tq, t40, y40, one_row(1.5), one_row(4.0),
                                     one_row(0.5), 27, "C13 single draw");
        // Repeated times (entries 0 and 3: 17.3; entries 2 and 8: 3.5) are
        // the same node, so f_star is identical there.
        bool same_rep = true;
        for (const auto* p : {&p13, &q13}) {
            const arma::mat& FS = p->at("f_star");
            for (std::size_t d = 0; d < FS.n_rows; ++d)
                same_rep = same_rep && FS(d, 0) == FS(d, 3) && FS(d, 2) == FS(d, 8);
        }
        // f_mean and f_sd against celerite_gp_block::predict_mean_var.
        double worst_cel = 0.0;
        const arma::mat& A = hist13.at("amp");
        const arma::mat& T = hist13.at("tau");
        const arma::mat& S = hist13.at("sigma");
        for (std::size_t d = 0; d < A.n_rows; ++d) {
            AI4BayesCode::celerite_gp_block cel(AI4BayesCode::celerite_gp_block_config{});
            AI4BayesCode::block_context ctx;
            ctx["t"] = t40;
            ctx["y"] = y40;
            ctx["a_real"] = arma::vec{A(d, 0) * A(d, 0)};
            ctx["c_real"] = arma::vec{1.0 / std::max(T(d, 0), 1e-10)};
            ctx["obs_diag"] = arma::vec{S(d, 0) * S(d, 0)};
            cel.set_context(ctx);
            std::mt19937_64 unused;
            cel.step(unused);
            const auto mv = cel.predict_mean_var(tq);
            for (std::size_t i = 0; i < tq.n_elem; ++i)
                worst_cel = std::max({worst_cel, rel(p13.at("f_mean")(d, i), mv.first[i]),
                                      rel(p13.at("f_sd")(d, i), std::sqrt(mv.second[i]))});
        }
        std::printf("C13 repeated times: f_star %s; max rel |(f_mean, f_sd) - "
                    "predict_mean_var| = %.3g\n",
                    same_rep ? "identical" : "DIFFERS", worst_cel);
        check(ok_h && ok_s && same_rep && worst_cel <= 1e-9,
              "C13 at unsorted, repeated and out-of-range new times f_star is one joint draw "
              "over all new times; f_mean, f_sd equal the dense and the celerite values");
    }

    // C14: mean and covariance of many joint draws at N = 40, in the
    // single-draw path at a fixed state and for every stored draw d of the
    // history path.
    {
        const std::size_t N14 = N40;
        const arma::vec& t14 = t40;
        const arma::vec& y14 = y40;
        // Largest |sample - exact| / standard error over the entries of the
        // mean and of the covariance of the rows of F (f_star) and Y
        // (y_rep), exact values m, C and C + sigma^2 I; se(S_ij) =
        // sqrt((C_ii C_jj + C_ij^2) / M). zi: the covariance statistic of
        // draws independent over time points (covariance diag(C)).
        struct zs { double zm_f, zc_f, zm_y, zc_y, zi, lag1; };
        auto mc_z = [&](const arma::mat& F, const arma::mat& Y, double amp, double tau,
                        double sig) {
            const dense_post_t q = dense_post(t14, t14, y14, amp, tau, sig);
            arma::mat Cy = q.C;
            Cy.diag() += sig * sig;
            const double Md = static_cast<double>(F.n_rows);
            zs z{0, 0, 0, 0, 0, 0};
            auto one = [&](const arma::mat& X, const arma::mat& C, double& zm, double& zc) {
                const arma::rowvec mean = arma::mean(X, 0);
                const arma::mat Sx = arma::cov(X);
                for (std::size_t i = 0; i < N14; ++i) {
                    zm = std::max(zm, std::abs(mean[i] - q.m[i]) / std::sqrt(C(i, i) / Md));
                    for (std::size_t j = 0; j < N14; ++j) {
                        const double se = std::sqrt((C(i, i) * C(j, j) + C(i, j) * C(i, j)) / Md);
                        zc = std::max(zc, std::abs(Sx(i, j) - C(i, j)) / se);
                    }
                }
            };
            one(F, q.C, z.zm_f, z.zc_f);
            one(Y, Cy, z.zm_y, z.zc_y);
            for (std::size_t i = 0; i < N14; ++i)
                for (std::size_t j = 0; j < N14; ++j)
                    if (i != j)
                        z.zi = std::max(z.zi, std::abs(q.C(i, j)) /
                                                  std::sqrt((q.C(i, i) * q.C(j, j) +
                                                             q.C(i, j) * q.C(i, j)) / Md));
            for (std::size_t i = 0; i + 1 < N14; ++i)
                z.lag1 += q.C(i, i + 1) / std::sqrt(q.C(i, i) * q.C(i + 1, i + 1));
            z.lag1 /= static_cast<double>(N14 - 1);
            return z;
        };
        bool ok = true;
        auto report = [&](const zs& z, const char* label) {
            std::printf("C14 %s: f_star max z mean %.2f, cov %.2f; y_rep max z mean %.2f, "
                        "cov %.2f (independent over time: cov z up to %.0f); mean "
                        "corr(f_i, f_i+1 | y) = %.3f\n",
                        label, z.zm_f, z.zc_f, z.zm_y, z.zc_y, z.zi, z.lag1);
            ok = ok && z.zm_f <= 6.0 && z.zc_f <= 6.0 && z.zm_y <= 6.0 && z.zc_y <= 6.0;
        };

        // Single-draw path, 20000 calls at (amp, tau, sigma) = (1.5, 4, 0.5).
        {
            const std::size_t M = 20000;
            GPTimeSeries m14(t14, y14, 29, false);
            m14.set_current(start40);
            arma::mat F(M, N14), Y(M, N14);
            for (std::size_t s = 0; s < M; ++s) {
                const auto p = m14.predict_at(AI4BayesCode::state_map{}, true);
                F.row(s) = p.at("f_star");
                Y.row(s) = p.at("y_rep");
            }
            report(mc_z(F, Y, 1.5, 4.0, 0.5), "single draw, 20000 draws");
        }
        // History path, 10000 calls; row d of every call is a draw for the
        // hyperparameters of stored draw d (3 sweeps from start40).
        {
            const std::size_t M = 10000;
            GPTimeSeries h14(t14, y14, 45, true);
            h14.set_current(start40);
            h14.step(3);
            const auto hist14 = h14.get_history();
            const std::size_t D = hist14.at("amp").n_rows;
            std::vector<arma::mat> F(D, arma::mat(M, N14)), Y(D, arma::mat(M, N14));
            for (std::size_t s = 0; s < M; ++s) {
                const auto p = h14.predict_at(AI4BayesCode::state_map{});
                for (std::size_t d = 0; d < D; ++d) {
                    F[d].row(s) = p.at("f_star").row(d);
                    Y[d].row(s) = p.at("y_rep").row(d);
                }
            }
            for (std::size_t d = 0; d < D; ++d) {
                const double amp = hist14.at("amp")(d, 0), tau = hist14.at("tau")(d, 0),
                             sig = hist14.at("sigma")(d, 0);
                char label[160];
                std::snprintf(label, sizeof(label),
                              "history draw %zu of %zu (amp %.3f, tau %.3f, sigma %.3f), "
                              "10000 draws", d + 1, D, amp, tau, sig);
                report(mc_z(F[d], Y[d], amp, tau, sig), label);
            }
        }
        check(ok, "C14 mean and covariance of joint f_star / y_rep draws match m_d, C_d and "
                  "C_d + sigma_d^2 I within 6 standard errors");
    }

    // C15: non-finite times.
    {
        GPTimeSeries m15(t, y, 31, true);
        m15.step(3);
        bool named = true;
        for (const arma::vec& bad : {arma::vec{1.0, std::nan("")},
                                     arma::vec{std::numeric_limits<double>::infinity()}}) {
            for (bool last : {false, true}) {
                AI4BayesCode::state_map nb;
                nb["t"] = bad;
                std::string msg;
                try { (void)m15.predict_at(nb, last); } catch (const std::exception& e) { msg = e.what(); }
                const std::string want = "entry " + std::to_string(bad.n_elem) + " of t";
                std::printf("C15 predict_at(t = %s, last_draw_only = %d): %s\n",
                            bad.n_elem == 2 ? "{1, NaN}" : "{Inf}", last ? 1 : 0,
                            msg.empty() ? "(no error)" : msg.c_str());
                named = named && msg.find(want) != std::string::npos;
            }
        }
        check(named, "C15 a non-finite time throws an error that names the entry");
    }

    // C16: tied training times.
    {
        arma::vec t16 = t40, y16 = y40;
        t16[5] = t16[4];
        t16[6] = t16[4];
        t16[31] = t16[30];
        GPTimeSeries h16(t16, y16, 33, true);
        h16.set_current(start40);
        h16.step(3);
        const auto hist16 = h16.get_history();
        print_draws("C16 stored draws", hist16);
        const auto p16 = h16.predict_at(AI4BayesCode::state_map{});
        const bool ok = check_rows(p16, t16, t16, y16, hist16.at("amp"), hist16.at("tau"),
                                   hist16.at("sigma"), 33, "C16 tied training times");
        const arma::mat& FS = p16.at("f_star");
        bool tied = true;
        for (std::size_t d = 0; d < FS.n_rows; ++d)
            tied = tied && FS(d, 4) == FS(d, 5) && FS(d, 4) == FS(d, 6) && FS(d, 30) == FS(d, 31);
        check(ok && tied, "C16 tied training times share one node: dense match, identical f_star");
    }

    // C17: extreme hyperparameters.
    {
        const std::size_t N17 = 50;
        arma::vec t17(N17), y17(N17);
        std::mt19937_64 g17(5u);
        std::normal_distribution<double> nd17(0.0, 1.0);
        for (std::size_t i = 0; i < N17; ++i) {
            t17[i] = 0.7 * static_cast<double>(i) + 0.1 * static_cast<double>(i % 3);
            y17[i] = std::cos(0.3 * t17[i]) + 0.3 * nd17(g17);
        }
        AI4BayesCode::state_map q17;
        q17["t"] = arma::vec{-5.0, 3.3, 3.3, 10.0, 40.0, 1e6};
        int n_ok = 0, n_all = 0;
        for (double amp : {1e-150, 1e-6, 1.0, 1e6, 1e150})
            for (double tau : {1e-9, 1e-3, 1.0, 1e3, 1e12})
                for (double sig : {1e-12, 1e-4, 1.0, 1e6}) {
                    GPTimeSeries m17(t17, y17, 3, false);
                    AI4BayesCode::state_map s17;
                    s17["amp"] = arma::vec{amp};
                    s17["tau"] = arma::vec{tau};
                    s17["sigma"] = arma::vec{sig};
                    m17.set_current(s17);
                    for (int k = 0; k < 2; ++k) {
                        const auto p = k ? m17.predict_at(q17, true)
                                         : m17.predict_at(AI4BayesCode::state_map{}, true);
                        bool ok = p.at("f_sd").min() >= 0.0;
                        for (const char* key : {"f_mean", "f_sd", "f_star", "y_rep"})
                            ok = ok && p.at(key).is_finite();
                        n_ok += ok;
                        ++n_all;
                    }
                }
        std::printf("C17 extreme (amp, tau, sigma): %d of %d calls give finite outputs, f_sd >= 0\n",
                    n_ok, n_all);
        check(n_ok == n_all, "C17 amp 1e-150..1e150, tau 1e-9..1e12, sigma 1e-12..1e6: finite "
                             "outputs and f_sd >= 0");
    }

    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "ALL PASSED", g_fail,
                g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}

// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later.
// ============================================================================
//  test_fix_examples_GPTimeSeries.cpp
//
//  examples/GPTimeSeries.cpp: celerite_logp (log p(y | amp, tau, sigma),
//  reported by get_current() as "logp"), predict_at at new times and at
//  the training times, and the predict DAG of get_dag().
//  Reference values are computed densely in this test: K = a exp(-c |dt|)
//  + (sigma^2 + 1e-10) I with a = amp^2, c = 1 / tau.
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
//        y_rep with one row per stored draw at the training t; f_mean_i and
//        f_sd_i^2 of draw d equal the dense GP posterior mean and marginal
//        variance of f(t_i) given y at (amp_d, tau_d, sigma_d) (relative
//        tolerance 1e-6, absolute 1e-8 * amp_d^2), and f_star = f_mean +
//        f_sd * z, y_rep = f_star + sigma_d * e bit for bit, with z, e drawn
//        from an engine seeded as predict_rng_ in the order z_1, e_1, z_2,
//        ..., i.e. each f_star_i from its marginal N(m_i, v_i), independently
//        over i;
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
//    C13 with N = 2 * kPredictBlock + 88 training times (three blocks),
//        predict_at({}) in history mode and in the single-draw path gives
//        f_mean and f_sd equal to one predict_mean_var call on all N times
//        (relative tolerance 1e-12; the number of bit-identical entries is
//        printed), and f_star and y_rep replay bit for bit from f_mean,
//        f_sd and an engine seeded as predict_rng_.
// ============================================================================
#include <armadillo>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <utility>

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

    // C7, C8: the posterior predictive at the training t.
    const std::uint64_t pred_xor = 0x9E3779B97F4A7C15ULL;   // predict_rng_ seed
    {
        GPTimeSeries h7(t, y, 13, true);
        h7.step(15);
        const auto hist7 = h7.get_history();
        const auto p7 = h7.predict_at(AI4BayesCode::state_map{});
        GPTimeSeries h8(t, y, 15, true);
        h8.step(10);
        const auto cur8 = h8.get_current();
        const auto p8 = h8.predict_at(AI4BayesCode::state_map{}, /*last_draw_only=*/true);

        // amp, tau, sigma of each row of p (hist rows, or the current state).
        auto run = [&](const AI4BayesCode::history_map& p, const arma::mat& A,
                       const arma::mat& T, const arma::mat& S, std::uint64_t seed,
                       const char* label) {
            const std::size_t nr = A.n_rows;
            bool shape = p.size() == 4;
            for (const char* k : {"f_mean", "f_sd", "f_star", "y_rep"})
                shape = shape && p.count(k) && p.at(k).n_rows == nr && p.at(k).n_cols == N;
            if (!shape) return false;
            const arma::mat& M = p.at("f_mean");
            const arma::mat& SD = p.at("f_sd");
            const arma::mat& FS = p.at("f_star");
            const arma::mat& YR = p.at("y_rep");
            double worst_m = 0.0, worst_v = 0.0;
            bool bits = true;
            std::mt19937_64 r(seed ^ pred_xor);
            std::normal_distribution<double> z01(0.0, 1.0);
            for (std::size_t d = 0; d < nr; ++d) {
                const arma::mat Kf = cov(t, t, A(d, 0), T(d, 0));
                arma::mat Ky = Kf;
                Ky.diag() += S(d, 0) * S(d, 0) + 1e-10;
                const arma::vec mu = Kf * arma::solve(Ky, y);
                const arma::mat KiK = arma::solve(Ky, Kf);
                const double k0 = A(d, 0) * A(d, 0);
                for (std::size_t i = 0; i < N; ++i) {
                    const double v = k0 - arma::dot(Kf.row(i), KiK.col(i));
                    worst_m = std::max(worst_m, rel(M(d, i), mu[i]));
                    worst_v = std::max(worst_v, std::abs(SD(d, i) * SD(d, i) - v) / k0);
                    const double fs = M(d, i) + SD(d, i) * z01(r);
                    const double yr = fs + S(d, 0) * z01(r);
                    const double fs_out = FS(d, i), yr_out = YR(d, i);
                    bits = bits && std::memcmp(&fs, &fs_out, sizeof(double)) == 0 &&
                           std::memcmp(&yr, &yr_out, sizeof(double)) == 0;
                }
            }
            std::printf("%s: %zu row(s); max rel |f_mean - dense| = %.3g, "
                        "max |f_sd^2 - dense var| / amp^2 = %.3g, replay %s\n",
                        label, nr, worst_m, worst_v, bits ? "bit-identical" : "DIFFERS");
            return worst_m <= 1e-6 && worst_v <= 1e-8 && bits;
        };
        check(run(p7, hist7.at("amp"), hist7.at("tau"), hist7.at("sigma"), 13, "C7 history"),
              "C7 history-mode predict_at({}) at the training t draws each f_star_i from the "
              "marginal N(m_i, v_i) of f(t_i) given y and theta_d, independently over i, and "
              "y_rep = f_star + sigma_d * e");
        check(run(p8, arma::mat(1, 1, arma::fill::value(cur8.at("amp")[0])),
                  arma::mat(1, 1, arma::fill::value(cur8.at("tau")[0])),
                  arma::mat(1, 1, arma::fill::value(cur8.at("sigma")[0])), 15, "C8 single draw"),
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

    // C13: three blocks of times at the training t.
    {
        const std::size_t N13 = 2 * kPredictBlock + 88;
        arma::vec t13(N13), y13(N13);
        std::mt19937_64 g13(73u);
        for (std::size_t i = 0; i < N13; ++i) {
            t13[i] = 0.5 * static_cast<double>(i);
            y13[i] = std::sin(0.05 * t13[i]) + 0.3 * nd(g13);
        }
        GPTimeSeries h13(t13, y13, 25, true);
        h13.step(4);
        const auto hist13 = h13.get_history();
        const auto p13 = h13.predict_at(AI4BayesCode::state_map{});
        GPTimeSeries s13(t13, y13, 27, false);
        s13.step(3);
        const auto cur13 = s13.get_current();
        const auto q13 = s13.predict_at(AI4BayesCode::state_map{}, /*last_draw_only=*/true);

        // m and v from ONE predict_mean_var call on all N13 times.
        auto one_call = [&](double amp, double tau, double sigma) {
            AI4BayesCode::celerite_gp_block cel(AI4BayesCode::celerite_gp_block_config{});
            AI4BayesCode::block_context ctx;
            ctx["t"] = t13;
            ctx["y"] = y13;
            ctx["a_real"] = arma::vec{amp * amp};
            ctx["c_real"] = arma::vec{1.0 / std::max(tau, 1e-10)};
            ctx["obs_diag"] = arma::vec{sigma * sigma};
            cel.set_context(ctx);
            std::mt19937_64 unused;
            cel.step(unused);
            return cel.predict_mean_var(t13);
        };
        auto run13 = [&](const AI4BayesCode::history_map& p, const arma::mat& A,
                         const arma::mat& T, const arma::mat& S, std::uint64_t seed,
                         const char* label) {
            const std::size_t nr = A.n_rows;
            bool shape = p.size() == 4;
            for (const char* k : {"f_mean", "f_sd", "f_star", "y_rep"})
                shape = shape && p.count(k) && p.at(k).n_rows == nr && p.at(k).n_cols == N13;
            if (!shape) return false;
            const arma::mat& M = p.at("f_mean");
            const arma::mat& SD = p.at("f_sd");
            const arma::mat& FS = p.at("f_star");
            const arma::mat& YR = p.at("y_rep");
            double worst = 0.0;
            std::size_t n_bits = 0;
            bool replay = true;
            std::mt19937_64 r(seed ^ pred_xor);
            std::normal_distribution<double> z01(0.0, 1.0);
            for (std::size_t d = 0; d < nr; ++d) {
                const auto mv = one_call(A(d, 0), T(d, 0), S(d, 0));
                for (std::size_t i = 0; i < N13; ++i) {
                    const double m = mv.first[i], sd = std::sqrt(std::max(mv.second[i], 0.0));
                    const double mo = M(d, i), so = SD(d, i);
                    worst = std::max({worst, rel(mo, m), rel(so, sd)});
                    n_bits += (std::memcmp(&mo, &m, sizeof(double)) == 0 &&
                               std::memcmp(&so, &sd, sizeof(double)) == 0);
                    const double fs = mo + so * z01(r);
                    const double yr = fs + S(d, 0) * z01(r);
                    const double fs_out = FS(d, i), yr_out = YR(d, i);
                    replay = replay && std::memcmp(&fs, &fs_out, sizeof(double)) == 0 &&
                             std::memcmp(&yr, &yr_out, sizeof(double)) == 0;
                }
            }
            std::printf("%s: %zu row(s) x %zu; max rel |(f_mean, f_sd) - one call| = %.3g, "
                        "%zu of %zu entries bit-identical; replay %s\n",
                        label, nr, N13, worst, n_bits, nr * N13,
                        replay ? "bit-identical" : "DIFFERS");
            return worst <= 1e-12 && replay;
        };
        const bool ok_h = run13(p13, hist13.at("amp"), hist13.at("tau"), hist13.at("sigma"), 25,
                                "C13 history");
        const bool ok_s = run13(q13, arma::mat(1, 1, arma::fill::value(cur13.at("amp")[0])),
                                arma::mat(1, 1, arma::fill::value(cur13.at("tau")[0])),
                                arma::mat(1, 1, arma::fill::value(cur13.at("sigma")[0])), 27,
                                "C13 single draw");
        check(ok_h && ok_s,
              "C13 predict_at({}) in blocks of kPredictBlock times equals one predict_mean_var "
              "call on all N times, draws replayed bit for bit");
    }

    std::printf("%s (%d failure%s)\n", g_fail ? "FAILED" : "ALL PASSED", g_fail,
                g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}

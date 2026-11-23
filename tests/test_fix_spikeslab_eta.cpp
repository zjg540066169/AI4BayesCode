/*================================================================================
 *  SpikeSlabRJMCMC: the derived key "eta" = X beta, and the freeze warning of
 *  gamma_beta_rj.
 *
 *  predict_at(list()) on the single-draw path samples
 *      y_rep_i = eta_i + sigma * z_i,   z_i ~ N(0, 1) from the predict RNG,
 *  with the stored eta, so eta must equal X beta of the current X and beta
 *  whenever predict_at can be called. Checks:
 *
 *    eta: y_rep of predict_at(list()) equals the value computed in this
 *        test from the current X, beta and sigma, with a copy of the predict
 *        RNG (seed rng_seed ^ 0x9E3779B97F4A7C15), relative tolerance 1e-12:
 *        at construction; after set_current of beta, of gamma and beta, of
 *        gamma alone (it zeroes beta), of X with the same and with a new row
 *        count, of y; after set_current calls that throw (after beta was
 *        written, before X was written, before beta was written); after
 *        sweeps; and through predict_at(list(), last_draw_only = true) of a
 *        model with keep_history = true. The stored "N" equals the row count
 *        of the stored X after each of these.
 *
 *    freeze warning: freeze("gamma_beta_rj") returns no warning that "sse"
 *        or "eta" are held stale: the model recomputes "sse" in step() before
 *        the first sweep after any set_current and "eta" in set_current.
 *
 *  The draws of step() do not depend on eta (no block's context contains
 *  it); the demo output of the example is unchanged.
 *================================================================================*/

#include <armadillo>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/kernel_control_mixin.hpp"

#include "portable_rng.hpp"

#define main spikeslab_demo_main
#include "../examples/SpikeSlabRJMCMC.cpp"
#undef main

// An explicit instantiation may name a private member, which gives the test
// access to the wrapper's composite_block.
template <typename Tag, typename Tag::type Member>
struct expose_member {
    friend typename Tag::type member_of(Tag) { return Member; }
};
struct impl_tag {
    using type = std::unique_ptr<AI4BayesCode::composite_block> SpikeSlabRJMCMC::*;
    friend type member_of(impl_tag);
};
template struct expose_member<impl_tag, &SpikeSlabRJMCMC::impl_>;

namespace {

int G_FAILED = 0;
void check(bool ok, const std::string& tag) {
    std::printf("  %s  %s\n", ok ? "PASS" : "FAIL", tag.c_str());
    if (!ok) ++G_FAILED;
}

AI4BayesCode::composite_block& impl(const SpikeSlabRJMCMC& m) {
    return *(m.*member_of(impl_tag{}));
}

void make_data(std::size_t N, std::size_t p, std::uint64_t seed,
               arma::mat& X, arma::vec& y) {
    std::mt19937_64 g(seed);
    X.set_size(N, p);
    for (std::size_t j = 0; j < p; ++j)
        for (std::size_t i = 0; i < N; ++i) X(i, j) = prng::n01(g);
    for (std::size_t j = 0; j < p; ++j) X.col(j) -= arma::mean(X.col(j));
    y.set_size(N);
    for (std::size_t i = 0; i < N; ++i)
        y[i] = 2.5 * X(i, 0) - 1.8 * X(i, p / 2) + prng::n01(g);
    y -= arma::mean(y);
}

arma::vec flat(const arma::mat& M) { return arma::vec(arma::vectorise(M)); }

// A model under test with its own copy of the predict RNG and of X.
struct tracked {
    SpikeSlabRJMCMC m;
    std::mt19937_64 prng_copy;
    arma::vec X;       // vectorised, column-major, as set
    std::size_t p;
    tracked(const arma::mat& X0, const arma::vec& y0, int seed, bool keep)
        : m(X0, y0, 1.0, 1.0, seed, keep),
          prng_copy(static_cast<std::uint64_t>(seed) ^ 0x9E3779B97F4A7C15ULL),
          X(flat(X0)), p(X0.n_cols) {}
};

// y_rep of predict_at(list()) against eta = X beta of the test's X and the
// model's current beta, sigma from get_current(), z from the RNG copy.
void check_predict(tracked& t, const std::string& what, bool last_draw_only = false) {
    const auto cur = t.m.get_current();
    const arma::vec& beta = cur.at("beta");
    const double sigma = cur.at("sigma")[0];
    const std::size_t N = t.X.n_elem / t.p;
    const AI4BayesCode::history_map out =
        last_draw_only ? t.m.predict_at(AI4BayesCode::state_map{}, true)
                       : t.m.predict_at(AI4BayesCode::state_map{});
    std::normal_distribution<double> norm(0.0, 1.0);
    bool ok = out.count("y_rep") && out.at("y_rep").n_rows == 1 &&
              out.at("y_rep").n_cols == N;
    double worst = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        double eta = 0.0;
        for (std::size_t j = 0; j < t.p; ++j) eta += t.X[i + j * N] * beta[j];
        const double expect = eta + sigma * norm(t.prng_copy);
        if (!ok) continue;
        const double got = out.at("y_rep")(0, i);
        const double err = std::abs(got - expect);
        worst = std::max(worst, err);
        if (!(err <= 1e-12 * (std::abs(eta) + std::abs(expect)) + 1e-300)) ok = false;
    }
    if (!ok) std::printf("    largest difference %.3g\n", worst);
    check(ok, "eta: predict_at(list()) uses X beta of the current X and beta: " + what);
    const double stored_N = impl(t.m).data().get("N")[0];
    check(stored_N == static_cast<double>(N),
          "eta: the stored N is the row count of X: " + what);
}

AI4BayesCode::state_map one(const char* k, const arma::vec& v) {
    AI4BayesCode::state_map sm;
    sm[k] = v;
    return sm;
}

void run(std::size_t N, std::size_t p, int seed) {
    std::printf("N = %zu, p = %zu, seed = %d\n", N, p, seed);
    arma::mat X, X2, X3, X4;
    arma::vec y, y2, y3, y4;
    const std::size_t N3 = N / 2 + 5;
    make_data(N, p, 11, X, y);
    make_data(N, p, 22, X2, y2);
    make_data(N3, p, 33, X3, y3);
    make_data(N + 9, p, 44, X4, y4);

    tracked t(X, y, seed, false);
    check_predict(t, "at construction");

    arma::vec b1(p);
    for (std::size_t j = 0; j < p; ++j) b1[j] = 0.7 - 0.25 * double(j);
    t.m.set_current(one("beta", b1));
    check_predict(t, "after set_current(beta)");

    arma::vec g2(p, arma::fill::ones), b2(p);
    for (std::size_t j = 0; j < p; ++j) b2[j] = -0.4 + 0.3 * double(j);
    {
        AI4BayesCode::state_map sm;
        sm["gamma"] = g2;
        sm["beta"] = b2;
        t.m.set_current(sm);
    }
    check_predict(t, "after set_current(gamma, beta)");

    arma::vec g3(p, arma::fill::zeros);
    g3[0] = 1.0;
    t.m.set_current(one("gamma", g3));
    check_predict(t, "after set_current(gamma), which zeroes beta");

    t.m.set_current(one("X", flat(X2)));
    t.X = flat(X2);
    check_predict(t, "after set_current(X), same row count");

    t.m.set_current(one("X", flat(X3)));
    t.X = flat(X3);
    check_predict(t, "after set_current(X) with a new row count");
    t.m.set_current(one("y", y3));
    check_predict(t, "after set_current(y) of the new length");

    {   // beta is written, then the sigma check throws
        AI4BayesCode::state_map sm;
        sm["gamma"] = g2;
        sm["beta"] = arma::vec(p, arma::fill::value(0.15));
        sm["sigma"] = arma::vec{-1.0};
        bool threw = false;
        try { t.m.set_current(sm); } catch (const std::exception&) { threw = true; }
        check(threw, "eta: set_current with sigma = -1 throws");
        check_predict(t, "after a set_current that throws after writing beta");
    }
    {   // X and y of different lengths in one call: throws before X is written
        AI4BayesCode::state_map sm;
        sm["X"] = flat(X4);
        sm["y"] = y4.subvec(0, N);   // N + 1 elements, X4 has N + 9 rows
        bool threw = false;
        try { t.m.set_current(sm); } catch (const std::exception&) { threw = true; }
        check(threw, "eta: set_current with X and y of different lengths throws");
        check_predict(t, "after a set_current that throws before writing X");
    }
    {   // beta of the wrong length: throws before beta is written
        bool threw = false;
        try { t.m.set_current(one("beta", arma::vec(p + 1, arma::fill::ones))); }
        catch (const std::exception&) { threw = true; }
        check(threw, "eta: set_current with beta of the wrong length throws");
        check_predict(t, "after a set_current that throws before writing beta");
    }

    t.m.step(5);
    check_predict(t, "after 5 sweeps");
    {
        AI4BayesCode::state_map sm;
        sm["X"] = flat(X);
        sm["y"] = y;
        t.m.set_current(sm);
        t.X = flat(X);
    }
    check_predict(t, "after set_current(X, y) back to the first data");
    t.m.step(3);
    check_predict(t, "after 3 more sweeps");

    // keep_history = true: the single-draw path of predict_at.
    tracked h(X, y, seed + 1, true);
    check_predict(h, "keep_history = TRUE, last_draw_only, at construction", true);
    h.m.set_current(one("beta", b1));
    check_predict(h, "keep_history = TRUE, last_draw_only, after set_current(beta)", true);

    // No stale-derived warning for gamma_beta_rj.
    tracked f(X, y, seed + 2, false);
    f.m.step(2);
    const std::vector<std::string> w =
        impl(f.m).freeze(std::vector<std::string>{"gamma_beta_rj"}, false);
    std::string all;
    for (const auto& s : w) all += (all.empty() ? "" : " | ") + s;
    check(w.empty(), "freeze warning: freeze(\"gamma_beta_rj\") returns no stale-derived warning"
                     + (all.empty() ? std::string() : ": " + all));
    f.m.set_current(one("X", flat(X2)));
    f.X = flat(X2);
    f.m.step(2);
    check_predict(f, "gamma_beta_rj frozen, after set_current(X) and 2 sweeps");
    f.m.set_current(one("beta", b1));
    check_predict(f, "gamma_beta_rj frozen, after set_current(beta)");
    impl(f.m).unfreeze_all();
}

void guarded(std::size_t N, std::size_t p, int seed) {
    try {
        run(N, p, seed);
    } catch (const std::exception& e) {
        std::printf("  FAIL  exception: %s\n", e.what());
        ++G_FAILED;
    }
}

} // namespace

int main() {
    std::printf("=== test_fix_spikeslab_eta ===\n");
    guarded(60, 5, 7);
    guarded(17, 1, 3);
    std::printf("%s (%d failed)\n", G_FAILED ? "FAILED" : "ALL PASSED", G_FAILED);
    return G_FAILED ? 1 : 0;
}

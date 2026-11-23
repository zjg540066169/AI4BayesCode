/*================================================================================
 *  SpikeSlabRJMCMC: SSE cache of the sigma slice block.
 *
 *  SSE = ||y - X beta||^2 enters sigma's density through the derived key
 *  "sse", which is computed once per sweep instead of at every slice
 *  evaluation. Checks:
 *
 *    C1  sigma's context contains "sse" and not X, and "sse" is refreshed
 *        after gamma_beta_rj (the block that writes beta) updates.
 *
 *    C2  the draws are bit-identical to those of a reference copy of the model
 *        in which "sse" is recomputed from the current y, X and beta before
 *        every sweep, across construction, step(n), set_current of every input
 *        (a call that throws after writing beta, X and y set in separate
 *        calls with step(0) after each call, a change of N) and move
 *        construction / assignment. A stale "sse" on any of these paths
 *        changes the next sigma draw.
 *
 *    C3  the cached "sse" equals ||y - X beta||^2 computed in this test from
 *        the model's current y, X and beta (relative tolerance 1e-12; the
 *        BLAS result may differ in the last bits from this loop), after the
 *        first sweep, after step(n), and after one sweep following each
 *        set_current path of C2 and C4.
 *
 *    C4  X and y of different lengths (set_current of X with a new row count
 *        and no y): step(0) runs, step(1) raises the error "X has .. rows but
 *        y has length ..", and the failed step leaves the state and the RNG
 *        unchanged. Covered with X shorter than y while sigma and
 *        gamma_beta_rj are frozen (the step then uses X only through "sse"),
 *        and with X longer than y and nothing frozen.
 *
 *  The model and its reference run in one process, so the C2 comparison
 *  holds on every platform and BLAS.
 *================================================================================*/

#include <armadillo>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
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

// An explicit instantiation may name a private member, which gives the
// reference model below access to the wrapper's composite_block.
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

bool has(const std::vector<std::string>& v, const char* k) {
    return std::find(v.begin(), v.end(), k) != v.end();
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

bool same_state(const SpikeSlabRJMCMC& a, const SpikeSlabRJMCMC& b) {
    const auto sa = a.get_current(), sb = b.get_current();
    for (const char* k : {"gamma", "beta", "sigma", "tau", "pi"}) {
        const arma::vec& u = sa.at(k);
        const arma::vec& v = sb.at(k);
        if (u.n_elem != v.n_elem ||
            std::memcmp(u.memptr(), v.memptr(), u.n_elem * sizeof(double)) != 0)
            return false;
    }
    return true;
}

// Recompute "sse" of model r from its current y, X and beta.
void refresh_sse(SpikeSlabRJMCMC& r) {
    impl(r).data().refresh_key("sse");
}

// C3: the cached "sse" of m equals ||y - X beta||^2 computed here, with a
// plain loop, from m's current y, X and beta.
bool sse_current(const SpikeSlabRJMCMC& m) {
    const AI4BayesCode::shared_data_t& d = impl(m).data();
    if (!d.has("sse")) return false;
    const arma::vec& y    = d.get("y");
    const arma::vec& X    = d.get("X");
    const arma::vec& beta = d.get("beta");
    const std::size_t N = y.n_elem, p = beta.n_elem;
    if (X.n_elem != N * p) return false;
    double sse = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        double r = y[i];
        for (std::size_t j = 0; j < p; ++j) r -= X[i + j * N] * beta[j];
        sse += r * r;
    }
    const double cached = d.get("sse")[0];
    const bool ok = std::abs(cached - sse) <= 1e-12 * std::abs(sse);
    if (!ok)
        std::printf("    cached sse %.17g, computed %.17g\n", cached, sse);
    return ok;
}

// n sweeps of the model under test (a) and of the reference (r), with "sse"
// of r recomputed before each sweep. Returns false at the first sweep whose
// state differs.
bool sweeps(SpikeSlabRJMCMC& a, SpikeSlabRJMCMC& r, int n) {
    for (int s = 0; s < n; ++s) {
        a.step(1);
        refresh_sse(r);
        r.step(1);
        if (!same_state(a, r)) return false;
    }
    return true;
}

// C2 over n sweeps, with C3 on a after the first of them.
void check_sweeps(SpikeSlabRJMCMC& a, SpikeSlabRJMCMC& r, int n,
                  const std::string& what, bool pre_ok = true) {
    bool ok = sweeps(a, r, 1);
    const bool c3 = sse_current(a);
    ok = ok && sweeps(a, r, n - 1);
    check(pre_ok && ok, "C2 " + what);
    check(c3, "C3 " + what);
}

void set_both(SpikeSlabRJMCMC& a, SpikeSlabRJMCMC& r,
              const AI4BayesCode::state_map& sm) {
    a.set_current(sm);
    r.set_current(sm);
}

AI4BayesCode::state_map gamma_beta(std::size_t p, double b0) {
    arma::vec g(p, arma::fill::zeros), b(p, arma::fill::zeros);
    for (std::size_t j = 0; j < p; j += 2) { g[j] = 1.0; b[j] = b0 - 0.3 * j; }
    AI4BayesCode::state_map sm;
    sm["gamma"] = g;
    sm["beta"]  = b;
    return sm;
}

void run_case(std::size_t N, std::size_t p, int seed) {
    std::printf("N = %zu, p = %zu, seed = %d\n", N, p, seed);
    arma::mat X, X2, X3;
    arma::vec y, y2, y3;
    const std::size_t N3 = N / 2 + 7;
    make_data(N, p, 101, X, y);
    make_data(N, p, 202, X2, y2);
    make_data(N3, p, 303, X3, y3);

    SpikeSlabRJMCMC a(X, y, 1.0, 1.0, seed, false);
    SpikeSlabRJMCMC r(X, y, 1.0, 1.0, seed, false);

    // C1
    const auto dag = a.get_dag();
    const auto& reads = dag.gibbs_reads.at("sigma");
    check(has(reads, "sse") && !has(reads, "X"),
          "C1 sigma's context contains sse, not X");
    check(dag.gibbs_invalidates.count("gamma_beta_rj") &&
          has(dag.gibbs_invalidates.at("gamma_beta_rj"), "sse"),
          "C1 sse is refreshed after gamma_beta_rj");

    // C2, C3
    check_sweeps(a, r, 200, "construction and step(1)");

    a.step(150);
    for (int s = 0; s < 150; ++s) { refresh_sse(r); r.step(1); }
    check(sse_current(a), "C3 step(n)");
    check(same_state(a, r) && sweeps(a, r, 20), "C2 step(n)");

    set_both(a, r, gamma_beta(p, 0.9));
    check_sweeps(a, r, 50, "set_current gamma, beta");

    { AI4BayesCode::state_map sm; sm["beta"] = arma::vec(p, arma::fill::value(0.4));
      set_both(a, r, sm); }
    check_sweeps(a, r, 50, "set_current beta");

    { AI4BayesCode::state_map sm; sm["X"] = flat(X2); set_both(a, r, sm); }
    check_sweeps(a, r, 50, "set_current X");

    { AI4BayesCode::state_map sm; sm["y"] = y2; set_both(a, r, sm); }
    check_sweeps(a, r, 50, "set_current y");

    { AI4BayesCode::state_map sm; sm["sigma"] = arma::vec{1.6};
      sm["tau"] = arma::vec{0.7}; sm["pi"] = arma::vec{0.4}; set_both(a, r, sm); }
    check_sweeps(a, r, 50, "set_current sigma, tau, pi");

    { AI4BayesCode::state_map sm = gamma_beta(p, -0.6);
      sm["X"] = flat(X3); sm["y"] = y3; set_both(a, r, sm); }
    check_sweeps(a, r, 50, "set_current gamma, beta, X, y with a new N");

    {   // step(0) between the calls and before the sweeps computes nothing
        AI4BayesCode::state_map sx; sx["X"] = flat(X); set_both(a, r, sx);
        a.step(0); r.step(0);
        AI4BayesCode::state_map sy; sy["y"] = y; set_both(a, r, sy);
        a.step(0); r.step(0);
    }
    check_sweeps(a, r, 50, "set_current X then y in separate calls, step(0) after each");

    {   // beta is written, then the y length check throws
        AI4BayesCode::state_map sm = gamma_beta(p, 1.3);
        sm["y"] = arma::vec(N + 1, arma::fill::ones);
        int thrown = 0;
        try { a.set_current(sm); } catch (const std::exception&) { ++thrown; }
        try { r.set_current(sm); } catch (const std::exception&) { ++thrown; }
        check_sweeps(a, r, 50, "set_current that throws after writing beta",
                     thrown == 2);
    }

    {   // move construction, then move assignment
        SpikeSlabRJMCMC a2(std::move(a)), r2(std::move(r));
        set_both(a2, r2, gamma_beta(p, 0.2));
        check_sweeps(a2, r2, 50, "move construction");
        SpikeSlabRJMCMC a3(X2, y2, 1.0, 1.0, seed + 1, false);
        SpikeSlabRJMCMC r3(X2, y2, 1.0, 1.0, seed + 1, false);
        a3 = std::move(a2);
        r3 = std::move(r2);
        AI4BayesCode::state_map sm; sm["y"] = y2; set_both(a3, r3, sm);
        check_sweeps(a3, r3, 50, "move assignment");
    }
}

std::string mismatch_message(std::size_t rows, std::size_t n) {
    char buf[160];
    std::snprintf(buf, sizeof buf,
                  "X has %zu rows but y has length %zu; set X and y together "
                  "before step()", rows, n);
    return buf;
}

// The message of the exception raised by m.step(n), or "" when none is raised.
std::string step_error(SpikeSlabRJMCMC& m, int n) {
    try { m.step(n); } catch (const std::exception& e) { return e.what(); }
    return "";
}

// n plain sweeps of a and r (each model with its own "sse" mechanism); false
// at the first sweep whose state differs.
bool plain_sweeps(SpikeSlabRJMCMC& a, SpikeSlabRJMCMC& r, int n) {
    for (int s = 0; s < n; ++s) {
        a.step(1);
        r.step(1);
        if (!same_state(a, r)) return false;
    }
    return true;
}

// C4. The same set_current calls are made on a and on the reference r, and
// the steps that fail on a are not made on r, so equal states afterwards show
// that a failed step changes neither the state nor the RNG.
void run_mismatch(std::size_t N, std::size_t p, int seed) {
    std::printf("N = %zu, p = %zu, seed = %d, X and y of different lengths\n",
                N, p, seed);
    arma::mat X, Xs, Xl;
    arma::vec y, ys, yl;
    const std::size_t Ns = N / 2, Nl = N + 17;
    make_data(N, p, 101, X, y);
    make_data(Ns, p, 404, Xs, ys);
    make_data(Nl, p, 505, Xl, yl);

    SpikeSlabRJMCMC a(X, y, 1.0, 1.0, seed, false);
    SpikeSlabRJMCMC r(X, y, 1.0, 1.0, seed, false);
    a.step(20);
    r.step(20);

    // X shorter than y, sigma and gamma_beta_rj frozen.
    const std::vector<std::string> both = {"sigma", "gamma_beta_rj"};
    impl(a).freeze(both, true);
    impl(r).freeze(both, true);
    { AI4BayesCode::state_map sx; sx["X"] = flat(Xs); set_both(a, r, sx); }
    check(step_error(a, 0).empty(),
          "C4 step(0) with X shorter than y (sigma, gamma_beta_rj frozen)");
    check(step_error(a, 1) == mismatch_message(Ns, N),
          "C4 step(1) raises the X/y length error, X shorter than y "
          "(sigma, gamma_beta_rj frozen)");
    { AI4BayesCode::state_map sy; sy["y"] = ys; set_both(a, r, sy); }
    bool ok = plain_sweeps(a, r, 1);
    const bool c3_short = sse_current(a);
    ok = ok && plain_sweeps(a, r, 9);
    impl(a).unfreeze_all();
    impl(r).unfreeze_all();
    ok = ok && plain_sweeps(a, r, 20);
    check(ok, "C4 the failed step changes neither state nor RNG (X shorter)");
    check(c3_short, "C3 after y of the new length is set (X shorter)");

    // X longer than y, nothing frozen.
    { AI4BayesCode::state_map sx; sx["X"] = flat(Xl); set_both(a, r, sx); }
    check(step_error(a, 1) == mismatch_message(Nl, Ns),
          "C4 step(1) raises the X/y length error, X longer than y");
    check(step_error(a, 3) == mismatch_message(Nl, Ns),
          "C4 step(3) raises it again while X and y differ in length");
    { AI4BayesCode::state_map sy; sy["y"] = yl; set_both(a, r, sy); }
    ok = plain_sweeps(a, r, 1);
    const bool c3_long = sse_current(a);
    ok = ok && plain_sweeps(a, r, 29);
    check(ok, "C4 the failed steps change neither state nor RNG (X longer)");
    check(c3_long, "C3 after y of the new length is set (X longer)");
}

void guarded(void (*fn)(std::size_t, std::size_t, int),
             std::size_t N, std::size_t p, int seed) {
    try {
        fn(N, p, seed);
    } catch (const std::exception& e) {
        std::printf("  FAIL  exception: %s\n", e.what());
        ++G_FAILED;
    }
}

} // namespace

int main() {
    std::printf("=== test_fix_slice_sse_cache ===\n");
    guarded(run_case, 300, 8, 7);
    guarded(run_case, 40, 3, 11);
    guarded(run_mismatch, 300, 8, 7);
    guarded(run_mismatch, 40, 3, 11);
    std::printf("%s (%d failed)\n", G_FAILED ? "FAILED" : "ALL PASSED", G_FAILED);
    return G_FAILED ? 1 : 0;
}

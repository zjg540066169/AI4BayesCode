// ============================================================================
// Regression test for the genBART caches:
//   - the leaf partition of each tree (one array of the N observation indices
//     grouped by leaf), built by the M-step and reused by the next sweep's
//     BIRTH / DEATH / CHANGE move (draw_tree_cached);
//   - the stored log(y_i) of gamma_shape_lik and log(yc_i), log(1 - yc_i) of
//     beta_lik (filled by prepare()).
// Every check compares the cached path with the uncached path BIT FOR BIT:
//   1. kernel: draw_tree_cached with a persistent cache against draw_tree,
//      run in lockstep on the same kernel RNG stream; the cached partition
//      equals partition_by_leaf after every call; and the move takes its
//      partition from a valid cache: a cache marked valid with an empty
//      partition changes at least one step relative to draw_tree;
//   2. genbart_model: every sweep is compared with the same sweep run on a
//      fresh copy of the model (a copy starts with invalid caches), across
//      set_X, set_tree, set_Y, set_offset, copy / move construction, copy /
//      move assignment and DART;
//   3. likelihoods: prepared against unprepared gamma_shape_lik / beta_lik
//      (log_f, score, obs_info, update_nuisance), with mismatched y,
//      out-of-range obs_i, special values and re-prepare; then whole chains,
//      including Y replaced through genbart_block::set_context.
// Part 3 checks that the stored logs give the same values as computed logs;
// whether log_f, score and update_nuisance take the stored values is checked
// only by instrumentation or timing, because the likelihoods expose no count
// of their log evaluations.
// Deterministic (bart_rng::set_seed); runs in a few seconds.
// ============================================================================
#include <armadillo>
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/genbart_block.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using genbart::genbart_model;
using genbart::rjmcmc_hypers;

static int fails = 0, checks = 0;
static void ok(bool cond, const char* what, long n) {
    ++checks; if (!cond) ++fails;
    std::printf("  [%s] %-64s (n=%ld)\n", cond ? "PASS" : "FAIL", what, n);
}

static bool same_bits(double a, double b) { return std::memcmp(&a, &b, sizeof a) == 0; }
static bool same_vec(const arma::vec& a, const arma::vec& b) {
    return a.n_elem == b.n_elem &&
           std::memcmp(a.memptr(), b.memptr(), sizeof(double) * a.n_elem) == 0;
}
static std::string tree_str(const genbart::tree& t) {
    std::ostringstream os; os.precision(17); os << t; return os.str();
}

// part lists, for every leaf of t, the observations partition_by_leaf routes
// to it (in the same order), and each of the N observations once.
static bool same_partition(const genbart::leaf_partition& part, genbart::tree& t,
                           const std::vector<double>& Xr, std::size_t N, std::size_t p,
                           genbart::xinfo& xi) {
    std::map<genbart::tree::tree_p, std::vector<std::size_t> > fresh;
    genbart::partition_by_leaf(t, Xr.data(), N, p, xi, fresh);
    genbart::tree::npv leaves;
    t.getbots(leaves);
    if (part.num_leaves() != leaves.size()) return false;
    std::size_t total = 0;
    for (genbart::tree::tree_p leaf : leaves) {
        const genbart::leaf_rows r = part.rows(leaf);
        const std::vector<std::size_t> got(r.begin(), r.end());
        const auto it = fresh.find(leaf);
        if (it == fresh.end() ? !got.empty() : got != it->second) return false;
        total += got.size();
    }
    return total == N;
}

// Data: X uniform on [0,1]^p, eta = smooth signal.
struct data_t { arma::mat X; arma::vec eta; };
static data_t make_X(std::size_t N, std::size_t p, unsigned seed) {
    std::mt19937_64 g(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    data_t d; d.X.set_size(N, p); d.eta.set_size(N);
    for (std::size_t i = 0; i < N; ++i) for (std::size_t j = 0; j < p; ++j) d.X(i, j) = u(g);
    for (std::size_t i = 0; i < N; ++i)
        d.eta[i] = std::sin(3.0 * d.X(i, 0) * d.X(i, 1 % p)) + (d.X(i, 2 % p) - 0.5);
    return d;
}
static arma::vec poisson_y(const arma::vec& eta, unsigned seed) {
    std::mt19937_64 g(seed); arma::vec y(eta.n_elem);
    for (arma::uword i = 0; i < eta.n_elem; ++i) {
        std::poisson_distribution<int> po(std::exp(0.8 * eta[i] + 0.3)); y[i] = po(g);
    }
    return y;
}
static arma::vec gamma_y(const arma::vec& eta, unsigned seed) {
    std::mt19937_64 g(seed); arma::vec y(eta.n_elem);
    for (arma::uword i = 0; i < eta.n_elem; ++i) {
        std::gamma_distribution<double> ga(std::exp(0.5 + 0.6 * eta[i]), 0.5);
        y[i] = std::max(ga(g), 1e-12);
    }
    return y;
}
static arma::vec beta_y(const arma::vec& eta, unsigned seed) {
    std::mt19937_64 g(seed); arma::vec y(eta.n_elem);
    for (arma::uword i = 0; i < eta.n_elem; ++i) {
        const double mu = 1.0 / (1.0 + std::exp(-eta[i]));
        std::gamma_distribution<double> ga(12.0 * mu, 1.0), gb(12.0 * (1.0 - mu), 1.0);
        const double a = ga(g), b = gb(g); y[i] = a / (a + b);
        if (i % 53 == 0) y[i] = 0.0;      // clipped to 1e-10
        if (i % 59 == 0) y[i] = 1.0;      // clipped to 1 - 1e-10
    }
    return y;
}

// ---------------------------------------------------------------------------
// 1. Kernel: draw_tree_cached (persistent cache) vs draw_tree (no cache).
// ---------------------------------------------------------------------------
static void test_kernel_lockstep() {
    std::printf("\n[1] draw_tree_cached vs draw_tree, lockstep\n");
    const std::size_t N = 160, p = 3;
    data_t d = make_X(N, p, 1);
    arma::vec y = poisson_y(d.eta, 2);
    std::vector<double> Xr(N * p), Y(y.begin(), y.end()), lam(N), fA(N), fB(N);
    for (std::size_t i = 0; i < N; ++i) for (std::size_t j = 0; j < p; ++j) Xr[i * p + j] = d.X(i, j);
    genbart::xinfo xi(p);
    for (std::size_t j = 0; j < p; ++j)
        for (int c = 1; c <= 24; ++c) xi[j].push_back(c / 25.0);
    rjmcmc_hypers h; h.sigma_mu = 0.3;
    genbart::lik::poisson_lik lik;
    genbart::arn gen;
    genbart::tree a, b;
    genbart::leaf_partition_cache cache;
    bart_rng::set_seed(101);
    long n = 0, bad = 0, part_bad = 0, invalid = 0, accepted = 0;
    for (int k = 0; k < 400; ++k) {
        for (std::size_t i = 0; i < N; ++i) lam[i] = 0.2 * std::sin(0.37 * k + 0.11 * i);
        const std::mt19937_64 e0 = bart_rng::engine();
        const genbart::move_result ra = genbart::draw_tree_cached(
            a, Xr.data(), Y.data(), lam.data(), N, p, xi, h, lik, gen, true, fA.data(), cache);
        const std::mt19937_64 e1 = bart_rng::engine();
        bart_rng::engine() = e0;
        const genbart::move_result rb = genbart::draw_tree(
            b, Xr.data(), Y.data(), lam.data(), N, p, xi, h, lik, gen, true, fB.data());
        ++n;
        if (!(bart_rng::engine() == e1) || ra.move != rb.move || ra.accepted != rb.accepted ||
            tree_str(a) != tree_str(b) ||
            std::memcmp(fA.data(), fB.data(), sizeof(double) * N) != 0) ++bad;
        accepted += ra.accepted ? 1 : 0;
        if (!cache.valid) ++invalid;
        if (!same_partition(cache.part, a, Xr, N, p, xi)) ++part_bad;
    }
    ok(bad == 0, "same move, acceptance, tree, f_tree and RNG stream", n);
    ok(invalid == 0 && part_bad == 0, "cache valid after each call and equal to a fresh partition", n);
    ok(accepted > 20, "structure moves were accepted (the leaves changed)", accepted);

    // The move takes the partition from a valid cache: with a cache marked
    // valid whose partition is empty (no observations in any leaf), the
    // move's proposal, acceptance or RNG draws differ from draw_tree's.
    {
        genbart::tree c = a, d = a;
        long differ = 0;
        for (int k = 0; k < 40; ++k) {
            genbart::leaf_partition_cache empty_valid;
            empty_valid.valid = true;
            const std::mt19937_64 e0 = bart_rng::engine();
            genbart::draw_tree_cached(c, Xr.data(), Y.data(), lam.data(), N, p, xi, h, lik, gen,
                                      true, fA.data(), empty_valid);
            const std::mt19937_64 e1 = bart_rng::engine();
            bart_rng::engine() = e0;
            genbart::draw_tree(d, Xr.data(), Y.data(), lam.data(), N, p, xi, h, lik, gen, true, fB.data());
            if (!(bart_rng::engine() == e1) || tree_str(c) != tree_str(d) ||
                std::memcmp(fA.data(), fB.data(), sizeof(double) * N) != 0) ++differ;
            d = c; bart_rng::engine() = e1;       // continue both from the cached step
        }
        ok(differ > 0, "a valid cache's partition is used by the move", differ);
    }

    // Without an M-step: valid only when f_tree_out is requested.
    std::vector<double> f(N);
    genbart::draw_tree_cached(a, Xr.data(), Y.data(), lam.data(), N, p, xi, h, lik, gen, false, f.data(), cache);
    const bool v1 = cache.valid && !cache.part.empty();
    genbart::leaf_partition_cache c2(cache), c3; c3 = cache;   // copies of a valid cache
    ok(!c2.valid && c2.part.empty() && !c3.valid && c3.part.empty(), "copies of a valid cache start empty and invalid", 2);
    genbart::draw_tree_cached(a, Xr.data(), Y.data(), lam.data(), N, p, xi, h, lik, gen, false, nullptr, cache);
    ok(v1 && !cache.valid, "no M-step: valid with f_tree_out, invalid without", 2);
}

// ---------------------------------------------------------------------------
// 2. genbart_model: each sweep equals the same sweep on a fresh copy.
// ---------------------------------------------------------------------------
template <class Lik>
struct sweep_checker {
    Lik& lik;
    genbart::arn gen;
    long n = 0, bad = 0;
    explicit sweep_checker(Lik& l) : lik(l) {}
    // One sweep of m, compared with the same sweep of an uncached copy of m.
    void sweep(genbart_model& m) {
        genbart_model ref(m);                  // copy: every cache invalid
        const Lik lik0 = lik;                  // likelihood state (shared pointer)
        const std::mt19937_64 e0 = bart_rng::engine();
        m.update_step(gen);
        const std::mt19937_64 e1 = bart_rng::engine();
        const Lik lik1 = lik;
        lik = lik0; bart_rng::engine() = e0;
        ref.update_step(gen);
        ++n;
        if (!(bart_rng::engine() == e1) || !same_vec(m.current_f_train(), ref.current_f_train()) ||
            m.get_tree() != ref.get_tree() ||
            !same_bits(m.current_sigma_mu(), ref.current_sigma_mu()) ||
            !same_bits(m.current_dart_theta(), ref.current_dart_theta()) ||
            !same_vec(m.current_var_probs(), ref.current_var_probs()) ||
            lik.nuisance_snapshot() != lik1.nuisance_snapshot()) ++bad;
        lik = lik1; bart_rng::engine() = e1;
    }
    void sweeps(genbart_model& m, int k) { for (int s = 0; s < k; ++s) sweep(m); }
};

template <class Lik>
static void test_model_paths(const char* name, Lik& lik, const arma::vec& y, const arma::vec& y2) {
    std::printf("\n[2] genbart_model (%s) vs uncached copy, every sweep\n", name);
    const std::size_t N = 180, p = 4, T = 12;
    data_t d = make_X(N, p, 3), d2 = make_X(N, p, 4);
    arma::vec off(N, arma::fill::zeros), off2 = 0.1 * d2.eta;
    rjmcmc_hypers h; h.dart_requested = true;
    bart_rng::set_seed(202);
    genbart_model m(d.X, y, off, &lik, h, T);
    sweep_checker<Lik> sc(lik);
    sc.sweeps(m, 30);                       ok(sc.bad == 0, "plain sweeps", sc.n);
    const std::string older = m.get_tree();
    long b0 = sc.bad, n0 = sc.n;
    m.set_X(d2.X);  sc.sweeps(m, 20);       ok(sc.bad == b0, "after set_X", sc.n - n0);
    b0 = sc.bad; n0 = sc.n;
    m.set_tree(m.get_tree()); sc.sweeps(m, 20);
    ok(sc.bad == b0, "after set_tree(get_tree())", sc.n - n0);
    b0 = sc.bad; n0 = sc.n;
    m.set_tree(older); sc.sweeps(m, 20);    ok(sc.bad == b0, "after set_tree(older forest)", sc.n - n0);
    b0 = sc.bad; n0 = sc.n;
    m.set_Y(y2); m.set_offset(off2); sc.sweeps(m, 20);
    ok(sc.bad == b0, "after set_Y and set_offset", sc.n - n0);
    b0 = sc.bad; n0 = sc.n;
    m.startdart(); sc.sweeps(m, 20);        ok(sc.bad == b0, "with DART active", sc.n - n0);

    genbart::arn gen;
    b0 = sc.bad; n0 = sc.n;
    genbart_model c(m);   sc.sweeps(c, 15); ok(sc.bad == b0, "copy-constructed model", sc.n - n0);
    b0 = sc.bad; n0 = sc.n;
    genbart_model a(m);   for (int s = 0; s < 3; ++s) a.update_step(gen);   // a holds valid caches
    a = c;                sc.sweeps(a, 15); ok(sc.bad == b0, "copy-assigned model", sc.n - n0);
    b0 = sc.bad; n0 = sc.n;
    genbart_model mv(std::move(c)); sc.sweeps(mv, 15);
    ok(sc.bad == b0, "move-constructed model", sc.n - n0);
    b0 = sc.bad; n0 = sc.n;
    genbart_model b(m);   for (int s = 0; s < 3; ++s) b.update_step(gen);
    b = std::move(a);     sc.sweeps(b, 15); ok(sc.bad == b0, "move-assigned model", sc.n - n0);
    b0 = sc.bad; n0 = sc.n;
    m = b;                sc.sweeps(m, 15); ok(sc.bad == b0, "model assigned over a live model", sc.n - n0);
}

// ---------------------------------------------------------------------------
// 3. Likelihood stores.
// ---------------------------------------------------------------------------
struct gamma_plain : genbart::lik::gamma_shape_lik {      // never stores
    using genbart::lik::gamma_shape_lik::gamma_shape_lik;
    void prepare(const std::vector<double>&) override {}
};
struct beta_plain : genbart::lik::beta_lik {
    using genbart::lik::beta_lik::beta_lik;
    void prepare(const std::vector<double>&) override {}
};

// log_f, score and obs_info of two likelihoods at (y[i], lam, obs[i]).
static long compare_calls(const genbart::likelihood& A, const genbart::likelihood& B,
                          const std::vector<double>& y, const std::vector<std::size_t>& obs) {
    const double lams[] = {-3.0, -0.4, 0.0, 0.25, 1.7, 9.0};
    long bad = 0;
    for (std::size_t k = 0; k < y.size(); ++k)
        for (double l : lams) {
            if (!same_bits(A.log_f(y[k], l, obs[k]), B.log_f(y[k], l, obs[k]))) ++bad;
            if (!same_bits(A.score(y[k], l, obs[k]), B.score(y[k], l, obs[k]))) ++bad;
            if (!same_bits(A.obs_info(y[k], l, obs[k]), B.obs_info(y[k], l, obs[k]))) ++bad;
        }
    return bad;
}

template <class Lik, class Plain>
static void test_lik_calls(const char* name, double nuis, std::vector<double> y) {
    std::printf("\n[3] %s: prepared vs unprepared, per call\n", name);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    y.push_back(0.0); y.push_back(-0.0); y.push_back(-2.0); y.push_back(nan); y.push_back(inf);
    y.push_back(1e-300); y.push_back(1.0); y.push_back(1.0 - 1e-12); y.push_back(1e-11);
    const std::size_t n = y.size();
    Lik A(nuis); Plain B(nuis);
    A.prepare(y);
    std::vector<std::size_t> obs(n), far(n);
    std::vector<double> y_other(n);
    for (std::size_t i = 0; i < n; ++i) { obs[i] = i; far[i] = n + 7 + i; y_other[i] = 0.5 * y[i] + 0.01; }
    ok(compare_calls(A, B, y, obs) == 0, "stored y at its own obs_i", (long)n);
    ok(compare_calls(A, B, y_other, obs) == 0, "different y at a stored obs_i", (long)n);
    ok(compare_calls(A, B, y, far) == 0, "obs_i beyond the store", (long)n);
    std::vector<double> y2(y.rbegin(), y.rend());
    A.prepare(y2);
    ok(compare_calls(A, B, y2, obs) == 0, "after prepare() with a new y", (long)n);
    ok(compare_calls(A, B, y, obs) == 0, "old y after the new prepare()", (long)n);
}

template <class Lik, class Plain>
static void test_lik_nuisance(const char* name, const arma::vec& yv) {
    std::printf("\n[3] %s: update_nuisance prepared vs unprepared\n", name);
    std::vector<double> y(yv.begin(), yv.end()), y2(y.size()), lam(y.size());
    for (std::size_t i = 0; i < y.size(); ++i) y2[i] = (i % 2) ? y[i] : 0.5 * y[i] + 0.25;
    Lik A(3.0); Plain B(3.0);
    A.prepare(y);
    genbart::arn gen;
    bart_rng::set_seed(303);
    long n = 0, bad = 0, moved = 0;
    for (int k = 0; k < 200; ++k) {
        for (std::size_t i = 0; i < y.size(); ++i) lam[i] = 0.3 * std::cos(0.21 * k + 0.07 * i);
        const std::vector<double>& yy = (k % 3 == 2) ? y2 : y;   // every 3rd call: other y
        const double before = A.nuisance_snapshot()[0];
        const std::mt19937_64 e0 = bart_rng::engine();
        A.update_nuisance(yy, lam, gen);
        const std::mt19937_64 e1 = bart_rng::engine();
        bart_rng::engine() = e0;
        B.update_nuisance(yy, lam, gen);
        ++n;
        if (!(bart_rng::engine() == e1) ||
            !same_bits(A.nuisance_snapshot()[0], B.nuisance_snapshot()[0])) ++bad;
        if (!same_bits(before, A.nuisance_snapshot()[0])) ++moved;
    }
    ok(bad == 0, "same nuisance draw and RNG stream", n);
    ok(moved > 10, "nuisance draws were accepted", moved);
}

// Whole chains through genbart_block and composite_block; the response is
// replaced through set_context (y_key) on odd sweeps and by set_Y once.
template <class Lik, class Plain>
static void test_lik_chain(const char* name, double nuis, const arma::vec& y, const arma::vec& y2) {
    std::printf("\n[3] %s: chain with stored vs computed logs\n", name);
    const std::size_t N = y.n_elem, p = 3;
    data_t d = make_X(N, p, 5);
    Lik la(nuis); Plain lb(nuis);
    auto build = [&](genbart::likelihood* lik, std::uint64_t seed) {
        bart_rng::set_seed(seed);
        auto comp = std::make_unique<AI4BayesCode::composite_block>("C");
        comp->data().set("yk", y);
        comp->data().set("r", arma::vec(N, arma::fill::zeros));
        comp->data().declare_dependencies("r", {"yk"});
        AI4BayesCode::genbart_block_config cfg;
        cfg.name = "r"; cfg.x_train = d.X; cfg.y_init = y; cfg.lik = lik; cfg.ntrees = 15;
        cfg.y_key = "yk";
        comp->add_child(std::make_unique<AI4BayesCode::genbart_block>(std::move(cfg)));
        return comp;
    };
    auto A = build(&la, 404); const std::mt19937_64 eA = bart_rng::engine();
    auto B = build(&lb, 404);
    auto& ba = dynamic_cast<AI4BayesCode::genbart_block&>(A->child(0));
    auto& bb = dynamic_cast<AI4BayesCode::genbart_block&>(B->child(0));
    std::mt19937_64 engA = eA, engB = bart_rng::engine(), ra(1), rb(1);
    long n = 0, bad = 0;
    for (int s = 0; s < 80; ++s) {
        const arma::vec& yy = (s % 2) ? y2 : y;
        A->data().set("yk", yy); B->data().set("yk", yy);
        if (s == 50) { ba.set_Y(y2); bb.set_Y(y2); }
        bart_rng::engine() = engA; A->step(ra); engA = bart_rng::engine();
        bart_rng::engine() = engB; B->step(rb); engB = bart_rng::engine();
        ++n;
        if (!same_vec(ba.current(), bb.current()) || ba.get_tree() != bb.get_tree() ||
            !same_bits(ba.current_sigma(), bb.current_sigma()) ||
            !same_bits(ba.current_sigma_mu(), bb.current_sigma_mu()) || !(engA == engB)) ++bad;
    }
    ok(bad == 0, "identical draws every sweep", n);
}

int main() {
    std::printf("genBART cache regression test (partition cache, log(y) stores)\n");
    test_kernel_lockstep();

    {
        data_t d = make_X(180, 4, 3);
        genbart::lik::poisson_lik lik;
        test_model_paths("poisson", lik, poisson_y(d.eta, 6), poisson_y(d.eta, 7));
    }
    {
        data_t d = make_X(180, 4, 3);
        arma::vec y = 2.0 * d.eta, y2 = 2.0 * d.eta;
        std::mt19937_64 g(8); std::normal_distribution<double> z(0.0, 0.5);
        for (arma::uword i = 0; i < y.n_elem; ++i) { y[i] += z(g); y2[i] += z(g); }
        genbart::lik::normal_lik lik(1.0, 3.0, 0.5);
        test_model_paths("normal", lik, y, y2);
    }
    {
        data_t d = make_X(180, 4, 3);
        genbart::lik::gamma_shape_lik lik(1.0);
        test_model_paths("gamma_shape", lik, gamma_y(d.eta, 9), gamma_y(d.eta, 10));
    }

    data_t d = make_X(120, 3, 11);
    {
        arma::vec y = gamma_y(d.eta, 12);
        test_lik_calls<genbart::lik::gamma_shape_lik, gamma_plain>(
            "gamma_shape_lik", 1.7, std::vector<double>(y.begin(), y.end()));
        test_lik_nuisance<genbart::lik::gamma_shape_lik, gamma_plain>("gamma_shape_lik", y);
        test_lik_chain<genbart::lik::gamma_shape_lik, gamma_plain>("gamma_shape_lik", 1.0, y, gamma_y(d.eta, 13));
    }
    {
        arma::vec y = beta_y(d.eta, 14);
        test_lik_calls<genbart::lik::beta_lik, beta_plain>(
            "beta_lik", 9.0, std::vector<double>(y.begin(), y.end()));
        test_lik_nuisance<genbart::lik::beta_lik, beta_plain>("beta_lik", y);
        test_lik_chain<genbart::lik::beta_lik, beta_plain>("beta_lik", 10.0, y, beta_y(d.eta, 15));
    }

    std::printf("\n%s  (%d checks, %d fail)\n",
                fails == 0 ? "[GENBART CACHE TEST PASS]" : "[GENBART CACHE TEST FAIL]", checks, fails);
    return fails == 0 ? 0 : 1;
}

/*
 * test_fix_vi_exact_cache.cpp
 *
 * In exact mode the categorical VI blocks tabulate log_density over the joint
 * state space once and evaluate it again only when set_context installs
 * different dependency values. This checks, for mean_field_categorical_vi_block
 * and structured_categorical_vi_block driven through composite_block with a
 * sibling that changes the dependencies at chosen sweeps:
 *
 *   1. every sweep's ELBO equals E_q[log p(z | beta, h)] + H(q) computed by
 *      direct enumeration with the dependency values of THAT sweep, so no
 *      sweep uses table values computed with earlier dependency values;
 *   2. log_density is evaluated for all states in the first sweep and in the
 *      sweeps whose dependencies changed, and not at all in the others;
 *   3. set_context called directly with the same bytes, a changed value, a
 *      removed key and a restored key triggers exactly the expected refills;
 *   4. a sweep after a dependency change makes no more operator new calls
 *      than a sweep without one (the refill writes into the table storage of
 *      the previous fill), and a copy of a block carries a valid table;
 *   5. a refill interrupted by a log_density exception is evaluated again in
 *      full at the next step;
 *   6. a set_context that throws after it has replaced one dependency value
 *      (std::bad_alloc from the insertion of a second key) makes the next
 *      step evaluate log_density again with the partly updated context.
 */

#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/shared_data.hpp"
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/mean_field_categorical_vi_block.hpp"
#include "AI4BayesCode/structured_categorical_vi_block.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <new>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

// Global operator new / delete replaced by counting versions over malloc /
// free. g_news counts operator new calls; when g_fail_next_new is set, the
// next operator new call throws std::bad_alloc. free is called through a
// volatile function pointer, so a delete inlined at a new site is not a
// direct new / free pair (GCC -Wmismatched-new-delete).
static std::size_t g_news = 0;
static bool g_fail_next_new = false;
static void free_block(void* p) noexcept { std::free(p); }
static void (*volatile g_free)(void*) noexcept = free_block;

void* operator new(std::size_t sz) {
    ++g_news;
    if (g_fail_next_new) { g_fail_next_new = false; throw std::bad_alloc(); }
    if (void* p = std::malloc(sz ? sz : 1)) return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t sz) { return ::operator new(sz); }
void operator delete(void* p) noexcept { g_free(p); }
void operator delete[](void* p) noexcept { g_free(p); }
void operator delete(void* p, std::size_t) noexcept { g_free(p); }
void operator delete[](void* p, std::size_t) noexcept { g_free(p); }

using namespace AI4BayesCode;

namespace {

int checks = 0, failures = 0;

void check(bool ok, const char* what) {
    ++checks;
    if (!ok) { ++failures; std::printf("  FAIL  %s\n", what); }
    else       std::printf("  ok    %s\n", what);
}

std::size_t g_calls = 0;
std::size_t g_throw_at = 0;   // nonzero: call number g_throw_at throws

const std::size_t n = 4, K = 3, total = 81;

// Potts chain: coupling beta on equal neighbours, field h_i on state 1.
// Z is arma::uvec (the block's argument) or std::vector (the enumeration).
template <class Z>
double chain_lp(const Z& z, double beta, const arma::vec& h) {
    double v = 0.0;
    for (std::size_t i = 0; i + 1 < n; ++i) if (z[i] == z[i + 1]) v += beta;
    for (std::size_t i = 0; i < n; ++i) if (z[i] == 1u) v += h[i];
    return v;
}

const arma::vec h_zero(n, arma::fill::zeros);

// beta defaults to 0.25 and h to zeros when the key is absent.
double ctx_lp(const arma::uvec& z, const block_context& ctx) {
    ++g_calls;
    if (g_throw_at != 0 && g_calls == g_throw_at)
        throw std::runtime_error("log_density failure");
    const double beta = ctx.count("beta") ? ctx.at("beta")[0] : 0.25;
    return chain_lp(z, beta, ctx.count("h") ? ctx.at("h") : h_zero);
}

// Sibling that sets beta (and h) to scheduled values at scheduled sweeps.
class schedule_block : public block_sampler {
public:
    schedule_block() { cur_ = arma::vec{0.8}; }
    void set_context(const block_context&) override {}
    void step(std::mt19937_64&) override {
        ++t_;
        if (t_ == 30)  cur_[0] = 1.4;
        if (t_ == 60)  cur_[0] = 0.3;
        if (t_ == 61)  cur_[0] = 0.3;   // rewritten with the same value
        if (t_ == 62)  cur_[0] = 2.0;
        if (t_ == 100) cur_[0] = 0.9;
    }
    const arma::vec& current() const override { return cur_; }
    void set_current(const arma::vec& v) override { cur_ = v; }
    std::size_t dim() const noexcept override { return 1; }
    const std::string& name() const noexcept override { return nm_; }
private:
    long t_ = 0; arma::vec cur_; std::string nm_ = "beta";
};

// E_q[log p] + H(q) by enumeration. q_of(z) returns q(z).
template <class QFn>
double exact_elbo(QFn q_of, double beta, const arma::vec& h, double entropy) {
    std::vector<std::size_t> z(n);
    double e = 0.0;
    for (std::size_t s = 0; s < total; ++s) {
        std::size_t r = s;
        for (std::size_t i = 0; i < n; ++i) { z[i] = r % K; r /= K; }
        e += q_of(z) * chain_lp(z, beta, h);
    }
    return e + entropy;
}

double entropy_of(const arma::vec& p) {
    double H = 0.0;
    for (double v : p) if (v > 0.0) H -= v * std::log(v);
    return H;
}

// q(z) for mean-field phi (node-major, K per node).
double q_mf(const arma::vec& phi, const std::vector<std::size_t>& z) {
    double q = 1.0;
    for (std::size_t i = 0; i < n; ++i) q *= phi[K * i + z[i]];
    return q;
}

// q(z) for cliques {0,1}, {2,3}: joint index z_a + K z_b.
double q_sc(const arma::vec& phi, const std::vector<std::size_t>& z) {
    return phi[z[0] + K * z[1]] * phi[K * K + z[2] + K * z[3]];
}

template <class Blk, class Cfg>
void run(const char* label, Cfg cfg, bool structured) {
    std::printf("%s\n", label);
    cfg.name = "z";
    cfg.cardinalities = arma::uvec(n); cfg.cardinalities.fill(K);
    cfg.log_density = ctx_lp;
    cfg.dependencies = {"beta", "h"};
    cfg.exact_enumeration = true;
    cfg.init_random_eps = 0.3;
    cfg.init_rng_seed = 3;

    composite_block comp("m");
    comp.data().set("beta", arma::vec{0.8});
    comp.data().set("h", arma::vec{0.4, -0.3, 0.2, 0.1});
    comp.data().set("z", arma::zeros(n));
    comp.add_child(std::make_unique<schedule_block>());
    auto b = std::make_unique<Blk>(cfg);
    Blk* vi = b.get();
    comp.add_child(std::move(b));
    comp.data().declare_dependencies("z", {"beta", "h"});

    // Sweeps whose dependency values differ from the previous sweep's.
    const std::set<long> refill = {1, 30, 60, 62, 80, 100};
    std::mt19937_64 rng(1);
    bool elbo_ok = true, calls_ok = true;
    int shown = 0;
    double worst = 0.0;
    for (long t = 1; t <= 150; ++t) {
        if (t == 80) comp.data().set("h", arma::vec{0.4, -0.3, 0.2, 0.6});
        const arma::vec phi = vi->current();
        g_calls = 0;
        comp.step(rng);
        const double beta = comp.data().get("beta")[0];
        const arma::vec h = comp.data().get("h");
        const double want = structured
            ? exact_elbo([&](const std::vector<std::size_t>& z) { return q_sc(phi, z); }, beta, h, entropy_of(phi))
            : exact_elbo([&](const std::vector<std::size_t>& z) { return q_mf(phi, z); }, beta, h, entropy_of(phi));
        const double err = std::abs(vi->current_elbo() - want) / (1.0 + std::abs(want));
        if (err > worst) worst = err;
        if (err > 1e-10) elbo_ok = false;
        const std::size_t expect = refill.count(t) ? total : 0;
        if (g_calls != expect) {
            calls_ok = false;
            if (shown++ < 3)
                std::printf("        sweep %ld: %zu log_density calls, expected %zu\n", t, g_calls, expect);
        }
    }
    std::printf("    largest relative ELBO error over 150 sweeps: %.2e\n", worst);
    check(!vi->converged(), "all 150 sweeps run the optimizer");
    check(elbo_ok, "each sweep's ELBO matches enumeration with that sweep's beta and h");
    check(calls_ok, "log_density runs over all states only when beta or h changed");

    // Direct set_context: same bytes, changed value, removed key, restored key.
    Blk d(cfg);
    block_context ctx;
    ctx["beta"] = arma::vec{0.8};
    ctx["h"] = arma::vec{0.4, -0.3, 0.2, 0.1};
    ctx["other"] = arma::vec{1.0};
    std::mt19937_64 r2(2);
    auto calls_for = [&](void (*edit)(block_context&)) {
        if (edit) edit(ctx);
        d.set_context(ctx);
        g_calls = 0;
        d.step(r2);
        return g_calls;
    };
    const std::size_t c0 = calls_for(nullptr);
    const std::size_t c1 = calls_for([](block_context& c) { c["beta"] = arma::vec{0.8}; });
    const std::size_t c2 = calls_for([](block_context& c) { c["other"] = arma::vec{2.0, 3.0}; });
    const std::size_t c3 = calls_for([](block_context& c) { c["beta"] = arma::vec{1.2}; });
    const std::size_t c4 = calls_for([](block_context& c) { c.erase("beta"); });
    const std::size_t c5 = calls_for([](block_context& c) { c["beta"] = arma::vec{1.2}; });
    const std::size_t c6 = calls_for([](block_context& c) { c["h"] = arma::vec{0.4, -0.3, 0.2, 0.1}; });
    const std::size_t c7 = calls_for([](block_context& c) { c["h"] = arma::vec{0.4, -0.3, 0.2, 0.7}; });
    std::printf("    direct set_context calls: %zu %zu %zu %zu %zu %zu %zu %zu\n",
                c0, c1, c2, c3, c4, c5, c6, c7);
    check(c0 == total && c1 == 0 && c2 == 0 && c3 == total && c4 == total && c5 == total &&
          c6 == 0 && c7 == total,
          "direct set_context: refill on first use and on every change only");

    // ELBO of the step just taken from phi (q before the step), beta and h.
    auto elbo_matches = [&](const Blk& b, const arma::vec& phi, double beta, const arma::vec& h) {
        const double want = structured
            ? exact_elbo([&](const std::vector<std::size_t>& z) { return q_sc(phi, z); }, beta, h, entropy_of(phi))
            : exact_elbo([&](const std::vector<std::size_t>& z) { return q_mf(phi, z); }, beta, h, entropy_of(phi));
        return std::abs(b.current_elbo() - want) / (1.0 + std::abs(want)) < 1e-10;
    };
    const arma::vec h1 = {0.4, -0.3, 0.2, 0.1};

    // 4. Operator new calls of a sweep with and without a dependency change.
    {
        Blk a(cfg);
        std::mt19937_64 r(4);
        block_context ctx_a, ctx_b;
        ctx_a["beta"] = arma::vec{0.8};  ctx_a["h"] = h1;
        ctx_b["beta"] = arma::vec{1.1};  ctx_b["h"] = h1;
        a.set_context(ctx_a);
        a.step(r);                                   // first fill
        auto sweep = [&](Blk& b, const block_context& c, std::size_t& news) {
            g_calls = 0;
            const std::size_t before = g_news;
            b.set_context(c);
            b.step(r);
            news = g_news - before;
            return g_calls;
        };
        std::size_t n_clean = 0, n_dirty = 0, n_back = 0, n_copy = 0;
        const std::size_t k_clean = sweep(a, ctx_a, n_clean);
        const std::size_t k_dirty = sweep(a, ctx_b, n_dirty);
        Blk copy(a);                                 // table valid for ctx_b
        const std::size_t k_back = sweep(a, ctx_a, n_back);
        const std::size_t k_copy = sweep(copy, ctx_b, n_copy);
        std::printf("    operator new calls per sweep: unchanged %zu, changed %zu, changed back %zu"
                    " (log_density calls %zu, %zu, %zu)\n",
                    n_clean, n_dirty, n_back, k_clean, k_dirty, k_back);
        check(k_clean == 0 && k_dirty == total && k_back == total,
              "a sweep after a dependency change evaluates every state once");
        check(n_dirty == n_clean && n_back == n_clean,
              "a sweep after a dependency change makes no more operator new calls than one without");
        check(k_copy == 0,
              "a copy taken after a fill uses the copied table while its context is unchanged");
    }

    // 5. log_density throws during a refill; the next step refills in full.
    {
        Blk e(cfg);
        std::mt19937_64 r(5);
        block_context ctx;
        ctx["beta"] = arma::vec{0.8};  ctx["h"] = h1;
        e.set_context(ctx);
        e.step(r);                                   // first fill
        ctx["beta"] = arma::vec{1.6};
        e.set_context(ctx);
        g_calls = 0;
        g_throw_at = 40;
        bool threw = false;
        try { e.step(r); } catch (const std::runtime_error&) { threw = true; }
        g_throw_at = 0;
        const arma::vec phi = e.current();
        g_calls = 0;
        e.step(r);
        std::printf("    after an exception at call 40 of a refill: %zu log_density calls\n", g_calls);
        check(threw && g_calls == total && elbo_matches(e, phi, 1.6, h1),
              "a refill interrupted by a log_density exception is evaluated again in full");
    }

    // 6. set_context replaces beta in place, then the insertion of h throws
    //    std::bad_alloc: the next step evaluates log_density with the new beta
    //    and no h.
    {
        Blk f(cfg);
        std::mt19937_64 r(6);
        block_context ctx_beta, ctx_full;
        ctx_beta["beta"] = arma::vec{0.8};
        ctx_full["beta"] = arma::vec{1.3};  ctx_full["h"] = h1;
        f.set_context(ctx_beta);                     // h absent
        f.step(r);                                   // fill with beta 0.8, h absent
        bool threw = false;
        g_fail_next_new = true;
        try { f.set_context(ctx_full); } catch (const std::bad_alloc&) { threw = true; }
        g_fail_next_new = false;
        arma::vec phi = f.current();
        g_calls = 0;
        f.step(r);
        const std::size_t k_partial = g_calls;
        const bool ok_partial = elbo_matches(f, phi, 1.3, h_zero);
        f.set_context(ctx_full);                     // now inserts h
        phi = f.current();
        g_calls = 0;
        f.step(r);
        const std::size_t k_full = g_calls;
        const bool ok_full = elbo_matches(f, phi, 1.3, h1);
        std::printf("    after set_context threw: %zu log_density calls; after the repeated"
                    " set_context: %zu\n", k_partial, k_full);
        check(threw && k_partial == total && ok_partial,
              "after a set_context exception the next step uses the partly updated context");
        check(k_full == total && ok_full,
              "the repeated set_context inserts h and the next step uses it");
    }
}

} // namespace

int main() {
    std::printf("=== test_fix_vi_exact_cache ===\n\n");
    run<mean_field_categorical_vi_block>("mean_field_categorical_vi_block, n=4 K=3 (81 states)",
                                         mean_field_categorical_vi_block_config{}, false);
    std::printf("\n");
    structured_categorical_vi_block_config sc;
    sc.clique_partition = {{0, 1}, {2, 3}};
    run<structured_categorical_vi_block>("structured_categorical_vi_block, cliques {0,1},{2,3}",
                                         sc, true);
    std::printf("\n%d checks, %d failures\n", checks, failures);
    if (failures) { std::printf("FAILED\n"); return 1; }
    std::printf("PASSED\n");
    return 0;
}

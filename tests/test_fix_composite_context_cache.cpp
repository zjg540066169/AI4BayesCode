// Regression test for how composite_block passes each child its context, and
// for composite_block::sync_child.
//
// composite_block::step() passes a child a fresh copy of its context
// (shared_data_t::build_context_for) when the values in the context take
// between composite_block::context_copy_min_bytes and context_copy_max_bytes
// bytes, and read-only views of the stored values
// (shared_data_t::context_view_for) otherwise; readapt_NUTS() passes views.
// This test sets the bounds to [24 KiB, 160 KiB] with
// AI4BAYESCODE_CONTEXT_COPY_MIN_BYTES / AI4BAYESCODE_CONTEXT_COPY_MAX_BYTES,
// so both ways are reached on every platform. Checked here:
//  - Every context passed to a probe child equals build_context_for() at that
//    moment: same keys, same iteration order, bitwise-equal values. Paths:
//    data().set (same size, new size, small values), write_back (by step and
//    directly, also with a new length in the same storage),
//    refresh_derived_for, refresh_key, refresh_all, run_stochastic_refresher,
//    composite set_current and set_context, declare_dependencies (drop,
//    reorder, duplicate, empty, missing key), freeze / unfreeze,
//    readapt_NUTS, add_child after sweeps, copy and move of the whole
//    shared_data_t, writes through a get_mut() reference, writes to caller
//    memory that a stored value aliases, nested composites, two children with
//    the same name. These sections run with contexts below the copy band
//    (views) and inside it (copies).
//  - Every context comes the way the size rule gives, and readapt_NUTS
//    passes views; both ways are reached on each side of both bounds; the
//    size is the sum over the context's values, a key listed twice counted
//    once, also after its length changes. A child whose dependency list is
//    declared after its first update gets the context of the list, below,
//    inside and above the band.
//  - A copy exists until the child's step, its write-back and the refresh
//    after it are done, and is freed after them.
//  - context_view_for() and context_view_or_copy_for() directly, after every
//    write path, with a copy made before the views; also for a block whose
//    dependency list is declared after the first call, and with a `copy`
//    argument that holds other entries.
//  - Steady sweeps allocate no buffer of the data's size below and above the
//    band (views), and exactly one per sweep inside it (the copy).
//  - The views are not made again for a copy when a value in the context
//    has new storage or a new length: context_view_or_copy_for() then calls
//    the global operator new as often as for a copy of unchanged values,
//    directly and in steady sweeps where a refresher returns a new vector in
//    every sweep.
//  - sync_child(i) after a loader called on child i gives the shared data,
//    current() and later draws that set_current(theta) gives, does not call
//    the child's set_current, and writes the named outputs of the overload
//    without an RNG.
// The members checked are detected at compile time, so the test also builds
// against trees without them, and then fails.

#include <cstddef>
#include <cstdlib>

#define AI4BAYESCODE_CONTEXT_COPY_MIN_BYTES (24 * 1024)
#define AI4BAYESCODE_CONTEXT_COPY_MAX_BYTES (160 * 1024)

// Count Armadillo allocations of at least g_big_threshold bytes, and record the
// addresses of those that are not freed yet (up to 64).
static std::size_t g_big_allocs = 0;
static std::size_t g_big_threshold = 0;  // 0: not counting
static void* g_big_live[64];
static int g_n_big_live = 0;
inline void* test_arma_alloc(std::size_t n) {
    void* p = std::malloc(n);
    if (g_big_threshold != 0 && n >= g_big_threshold) {
        ++g_big_allocs;
        if (g_n_big_live < 64) g_big_live[g_n_big_live++] = p;
    }
    return p;
}
inline void test_arma_free(void* p) {
    for (int i = 0; i < g_n_big_live; ++i) {
        if (g_big_live[i] == p) {
            g_big_live[i] = g_big_live[--g_n_big_live];
            break;
        }
    }
    std::free(p);
}
static bool big_live(const void* p) {
    for (int i = 0; i < g_n_big_live; ++i) if (g_big_live[i] == p) return true;
    return false;
}
#define ARMA_ALIEN_MEM_ALLOC_FUNCTION test_arma_alloc
#define ARMA_ALIEN_MEM_FREE_FUNCTION  test_arma_free

// Count calls of the global operator new while g_count_news is set (the
// containers of the context code allocate through it; Armadillo allocates
// through test_arma_alloc above). Not inlined, so GCC does not pair the
// free() below with the operator new of a caller (-Wmismatched-new-delete).
#include <new>
#if defined(__GNUC__)
#  define TEST_NOINLINE __attribute__((noinline))
#else
#  define TEST_NOINLINE
#endif
static std::size_t g_news = 0;
static bool g_count_news = false;
TEST_NOINLINE void* operator new(std::size_t n) {
    if (g_count_news) ++g_news;
    if (void* p = std::malloc(n == 0 ? 1 : n)) return p;
    throw std::bad_alloc();
}
TEST_NOINLINE void operator delete(void* p) noexcept { std::free(p); }
TEST_NOINLINE void operator delete(void* p, std::size_t) noexcept { std::free(p); }

#include <armadillo>
#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/shared_data.hpp"
#include "AI4BayesCode/composite_block.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using AI4BayesCode::block_context;
using AI4BayesCode::block_sampler;
using AI4BayesCode::composite_block;
using AI4BayesCode::shared_data_t;
using AI4BayesCode::state_map;

static int g_checks = 0, g_fails = 0;
static std::uint64_t g_hash = 1469598103934665603ULL;
// Off in the sections that run on trees with the size rule only, so the hash
// printed at the end can be compared between trees.
static bool g_hashing = true;

static void hash_bytes(const void* p, std::size_t n) {
    if (!g_hashing) return;
    const unsigned char* c = static_cast<const unsigned char*>(p);
    for (std::size_t i = 0; i < n; ++i) { g_hash ^= c[i]; g_hash *= 1099511628211ULL; }
}

static void ok(bool cond, const std::string& what) {
    ++g_checks;
    if (!cond) { ++g_fails; std::printf("  [FAIL] %s\n", what.c_str()); }
}

// ---- The size rule, if the tree has one -------------------------------------

template <class C, class = void>
struct has_copy_band : std::false_type {};
template <class C>
struct has_copy_band<C, std::void_t<decltype(C::context_copy_min_bytes),
                                    decltype(C::context_copy_max_bytes)>>
    : std::true_type {};

template <class C>
static std::size_t band_min() {
    if constexpr (has_copy_band<C>::value) return C::context_copy_min_bytes;
    else return 0;
}
template <class C>
static std::size_t band_max() {
    if constexpr (has_copy_band<C>::value) return C::context_copy_max_bytes;
    else return 0;
}
static constexpr bool k_has_rule = has_copy_band<composite_block>::value;

// Region of a context size: 0 below the band, 1 inside, 2 above.
static int region_of(std::size_t bytes) {
    if (bytes < band_min<composite_block>()) return 0;
    return bytes <= band_max<composite_block>() ? 1 : 2;
}

// g_ways[region][way]: number of contexts passed by step(), way 0 = a copy,
// 1 = views. g_readapt_ways[way]: the same for readapt_NUTS.
static int g_ways[3][2] = {{0, 0}, {0, 0}, {0, 0}};
static int g_readapt_ways[2] = {0, 0};
static bool g_in_readapt = false;  // set around readapt_NUTS calls

static std::size_t context_bytes(const block_context& ctx) {
    std::size_t n = 0;
    for (const auto& kv : ctx) n += kv.second.n_elem;
    return n * sizeof(double);
}

// Same keys in the same iteration order, bitwise-equal values.
static bool same_context(const block_context& a, const block_context& b) {
    if (a.size() != b.size()) return false;
    auto ia = a.begin();
    auto ib = b.begin();
    for (; ia != a.end(); ++ia, ++ib) {
        if (ia->first != ib->first) return false;
        const arma::vec& x = ia->second;
        const arma::vec& y = ib->second;
        if (x.n_rows != y.n_rows || x.n_cols != y.n_cols) return false;
        if (x.n_elem > 0 &&
            std::memcmp(x.memptr(), y.memptr(), x.n_elem * sizeof(double)) != 0)
            return false;
    }
    return true;
}

// 1 if every non-empty value in ctx is a view of the stored value, 0 if none
// is, -1 if mixed or nothing to tell.
static int view_state(const block_context& ctx, const shared_data_t& d) {
    int views = 0, copies = 0;
    for (const auto& kv : ctx) {
        if (kv.second.n_elem == 0) continue;
        if (kv.second.memptr() == d.get(kv.first).memptr()) ++views; else ++copies;
    }
    if (views + copies == 0) return -1;
    if (copies == 0) return 1;
    return views == 0 ? 0 : -1;
}

// Checks the way a context came against the size rule (views for
// readapt_NUTS) and counts it.
static void check_way(const block_context& ctx, int way, const std::string& who) {
    if (way < 0 || !k_has_rule) return;  // a missing rule is reported in main()
    const std::size_t bytes = context_bytes(ctx);
    int expect = 1;
    if (g_in_readapt) {
        ++g_readapt_ways[way];
    } else {
        const int region = region_of(bytes);
        ++g_ways[region][way];
        expect = region == 1 ? 0 : 1;
    }
    if (way != expect) {
        ok(false, who + ": a context of " + std::to_string(bytes) + " bytes came as "
                  + (way ? "views" : "a copy") + (g_in_readapt ? " in readapt_NUTS" : "")
                  + ", the rule gives " + (expect ? "views" : "a copy"));
    } else {
        ++g_checks;
    }
}

// Child that checks every context passed to it against build_context_for()
// on the composite's shared data and against the size rule. Its draw depends
// on every value passed to it and on the iteration order of the context.
class probe_block : public block_sampler {
public:
    probe_block(std::string name, std::size_t dim, const shared_data_t* watched,
                bool readapt = false)
        : name_(std::move(name)), cur_(dim, arma::fill::zeros),
          watched_(watched), readapt_(readapt) {}

    void set_context(const block_context& ctx) override {
        const block_context fresh = watched_->build_context_for(name_);
        ok(same_context(ctx, fresh),
           "child '" + name_ + "': context equals build_context_for");
        const int way = view_state(ctx, *watched_);
        modes_.push_back(way);
        check_way(ctx, way, "child '" + name_ + "'");
        acc_ = 0.0;
        std::size_t k = 0;
        for (const auto& kv : ctx) {
            ++k;
            hash_bytes(kv.first.data(), kv.first.size());
            hash_bytes(kv.second.memptr(), kv.second.n_elem * sizeof(double));
            for (std::size_t i = 0; i < kv.second.n_elem; ++i)
                acc_ += kv.second[i] * double(k) / double(1 + i % 7);
        }
        context_ = ctx;  // copy, per the block_sampler contract
        ++n_set_;
    }
    void step(std::mt19937_64& rng) override {
        std::normal_distribution<double> nd(0.0, 1.0);
        for (std::size_t j = 0; j < cur_.n_elem; ++j)
            cur_[j] = std::tanh(1e-3 * acc_) + 0.1 * nd(rng);
        hash_bytes(cur_.memptr(), cur_.n_elem * sizeof(double));
    }
    const arma::vec& current() const override { return cur_; }
    void set_current(const arma::vec& t) override { cur_ = t; ++n_set_current_; }
    std::size_t dim() const noexcept override { return cur_.n_elem; }
    const std::string& name() const noexcept override { return name_; }
    bool supports_readapt() const noexcept override { return readapt_; }
    void readapt(std::size_t, bool, std::mt19937_64&, std::size_t, double) override {
        ++n_readapt_;
    }

    int n_set_ = 0, n_readapt_ = 0, n_set_current_ = 0;
    std::vector<int> modes_;  // view_state of every context passed

private:
    std::string             name_;
    arma::vec               cur_;
    const shared_data_t*    watched_;
    bool                    readapt_;
    double                  acc_ = 0.0;
    block_context           context_;
};

static void readapt(composite_block& c, std::mt19937_64& rng, std::size_t n) {
    g_in_readapt = true;
    c.readapt_NUTS(n, false, rng);
    g_in_readapt = false;
}

// Child that uses only the lengths of its context (copies nothing).
class length_block : public block_sampler {
public:
    explicit length_block(std::string name)
        : name_(std::move(name)), cur_(1, arma::fill::zeros) {}
    void set_context(const block_context& ctx) override {
        n_ = 0;
        for (const auto& kv : ctx) n_ += kv.second.n_elem;
        first_ = ctx.count("A") ? ctx.at("A")[0] : 0.0;
    }
    void step(std::mt19937_64& rng) override {
        std::normal_distribution<double> nd(0.0, 1.0);
        cur_[0] = double(n_ % 11) + first_ + nd(rng);
    }
    const arma::vec& current() const override { return cur_; }
    void set_current(const arma::vec& t) override { cur_ = t; }
    std::size_t dim() const noexcept override { return 1; }
    const std::string& name() const noexcept override { return name_; }
    double first_ = 0.0;

private:
    std::string name_;
    arma::vec   cur_;
    std::size_t n_ = 0;
};

static arma::vec filled(std::size_t n, double a, double b) {
    arma::vec v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = a + b * std::sin(double(i) + a);
    return v;
}

static void sweeps(composite_block& c, std::mt19937_64& rng, int n) {
    for (int s = 0; s < n; ++s) c.step(rng);
}

// Fixed data X (200 * s elements), y, c; children pa, pb, pc; derived r
// (invalidated by pa) and q (refreshed only by refresh_key / refresh_all);
// stochastic eps. Returns pc (readapt-capable).
static probe_block* build_model(composite_block& comp, std::size_t s) {
    shared_data_t& d = comp.data();
    d.set("X", filled(200 * s, 0.5, 1.0));
    d.set("y", filled(30, -0.2, 0.7));
    d.set("c", arma::vec{2.0});
    d.set("eps", arma::vec(3, arma::fill::zeros));
    d.set("r", arma::vec(30, arma::fill::zeros));
    d.set("q", arma::vec(30, arma::fill::zeros));
    d.declare_dependencies("pa", {"X", "pb", "y"});
    d.declare_dependencies("pb", {"r", "pa", "X", "eps"});
    d.declare_dependencies("pc", {"c", "y", "q"});
    d.register_refresher("r", [](const shared_data_t& sd) -> arma::vec {
        return sd.get("y") * sd.get("pa")[0];
    });
    d.register_refresher("q", [](const shared_data_t& sd) -> arma::vec {
        return 2.0 * sd.get("y");
    });
    d.declare_invalidates("pa", {"r"});
    d.register_stochastic_refresher("eps",
        [](const shared_data_t& sd, std::mt19937_64& g) -> arma::vec {
            std::normal_distribution<double> nd(0.0, 1.0);
            arma::vec e(3);
            for (auto& x : e) x = nd(g) + sd.get("c")[0];
            return e;
        });
    auto pc_own = std::make_unique<probe_block>("pc", 1, &d, /*readapt=*/true);
    probe_block* pc = pc_own.get();
    comp.add_child(std::make_unique<probe_block>("pa", 1, &d));
    comp.add_child(std::make_unique<probe_block>("pb", 2, &d));
    comp.add_child(std::move(pc_own));
    d.refresh_all();
    return pc;
}

// ---- 1. Every ordinary write path; X has 200 * s elements, later 350 * s,
//         480 * s and 500 * s.
static void write_paths(std::mt19937_64& rng, std::size_t s) {
    composite_block comp("outer");
    probe_block* pc = build_model(comp, s);
    shared_data_t& d = comp.data();

    sweeps(comp, rng, 5);
    d.set("X", filled(200 * s, 1.5, 0.3));          // same size
    sweeps(comp, rng, 3);
    d.set("X", filled(500 * s, -1.0, 0.2));         // new size
    sweeps(comp, rng, 3);
    d.set("y", filled(30, 0.9, 0.1));               // q is now out of date
    sweeps(comp, rng, 2);
    d.refresh_key("q");
    sweeps(comp, rng, 2);
    d.set("y", filled(30, 0.35, 0.2));
    sweeps(comp, rng, 1);
    d.refresh_all();
    sweeps(comp, rng, 2);
    {
        std::mt19937_64 g2(5);
        d.run_stochastic_refresher("eps", g2);
    }
    sweeps(comp, rng, 2);
    d.write_back("c", arma::vec{3.0});
    sweeps(comp, rng, 2);
    d.set("pa", arma::vec{0.25});
    d.refresh_derived_for("pa");
    sweeps(comp, rng, 2);
    comp.set_current(arma::vec{0.1, -0.2, 0.3, 0.4});
    sweeps(comp, rng, 2);
    {
        block_context in;
        in["c"] = arma::vec{4.0};
        in["y"] = filled(30, 0.3, 0.3);
        comp.set_context(in);
    }
    sweeps(comp, rng, 2);

    // Dependency list changes: drop, reorder + add, empty, duplicate.
    d.declare_dependencies("pa", {"X", "y"});
    sweeps(comp, rng, 2);
    d.declare_dependencies("pa", {"y", "X", "pb"});
    sweeps(comp, rng, 2);
    d.declare_dependencies("pa", {});
    sweeps(comp, rng, 2);
    d.declare_dependencies("pa", {"X", "y", "X"});
    sweeps(comp, rng, 2);
    d.declare_dependencies("pa", {"X", "pb", "y"});
    sweeps(comp, rng, 2);

    // A frozen child is skipped; after unfreeze its context must include
    // the writes made while it was frozen.
    comp.freeze(std::vector<std::string>{"pc"}, /*quiet=*/true);
    const int pc_before = pc->n_set_;
    d.set("c", arma::vec{5.0});
    sweeps(comp, rng, 2);
    ok(pc->n_set_ == pc_before, "no context is passed to a frozen child");
    comp.unfreeze(std::vector<std::string>{"pc"});
    sweeps(comp, rng, 2);

    // readapt_NUTS passes the context too. pc's context is extended by X
    // here, so that it lies in the copy band when s puts X there.
    d.set("y", filled(30, -0.4, 0.5));
    d.declare_dependencies("pc", {"c", "y", "q", "X"});
    readapt(comp, rng, 3);
    ok(pc->n_readapt_ == 1, "readapt reached the readapt-capable child");
    sweeps(comp, rng, 1);
    d.declare_dependencies("pc", {"c", "y", "q"});

    // Whole-object copy and move assignment of the shared data.
    shared_data_t snap = comp.data();
    sweeps(comp, rng, 3);
    d.set("X", filled(500 * s, 2.0, 0.4));
    d.set("c", arma::vec{6.0});
    sweeps(comp, rng, 2);
    comp.data() = snap;                             // back to the older state
    sweeps(comp, rng, 3);
    shared_data_t snap2 = comp.data();
    d.set("y", filled(30, 1.1, 0.2));
    sweeps(comp, rng, 1);
    snap2.set("c", arma::vec{7.0});
    comp.data() = std::move(snap2);
    sweeps(comp, rng, 2);

    // A child added after sweeps, then a missing key.
    d.declare_dependencies("pd", {"X", "pa"});
    comp.add_child(std::make_unique<probe_block>("pd", 1, &d));
    sweeps(comp, rng, 2);
    d.declare_dependencies("pd", {"X", "nope"});
    std::string expected_msg, got_msg;
    try { (void)d.build_context_for("pd"); }
    catch (const std::out_of_range& e) { expected_msg = e.what(); }
    try { comp.step(rng); }
    catch (const std::out_of_range& e) { got_msg = e.what(); }
    ok(!expected_msg.empty() && got_msg == expected_msg,
       "missing key throws the build_context_for error");
    d.set("nope", arma::vec{1.0});
    sweeps(comp, rng, 30);
    d.set("X", filled(500 * s, 0.6, 0.6));
    d.write_back("c", arma::vec{8.0});
    sweeps(comp, rng, 3);
    // New lengths in the same storage: X shrinks and grows within its
    // buffer, c changes length within Armadillo's local buffer.
    d.write_back("X", filled(350 * s, 0.4, 0.2));
    d.write_back("c", arma::vec{1.0, 2.0});
    sweeps(comp, rng, 2);
    d.write_back("X", filled(480 * s, 0.1, 0.5));
    d.write_back("c", arma::vec{9.0});
    sweeps(comp, rng, 2);
}

// ---- 2. Writes through a held get_mut() reference ---------------------------
static void get_mut_writes(std::mt19937_64& rng, std::size_t s) {
    composite_block comp("mut");
    build_model(comp, s);
    shared_data_t& d = comp.data();
    sweeps(comp, rng, 3);
    arma::vec& Xref = d.get_mut("X");
    sweeps(comp, rng, 1);
    Xref[0] += 1.0;                                 // after the reference was taken
    sweeps(comp, rng, 1);
    Xref[1] -= 2.0;
    sweeps(comp, rng, 25);
    Xref[2] += 3.0;
    sweeps(comp, rng, 2);
}

// ---- 3. A get_mut() reference into an object later moved in -----------------
static void moved_in(std::mt19937_64& rng, std::size_t s) {
    composite_block comp("moved");
    build_model(comp, s);
    sweeps(comp, rng, 3);
    shared_data_t other = comp.data();
    arma::vec& yref = other.get_mut("y");
    comp.data() = std::move(other);                 // yref now refers into comp
    sweeps(comp, rng, 2);
    yref[0] += 1.0;
    sweeps(comp, rng, 1);
    yref[5] -= 0.5;
    sweeps(comp, rng, 2);
}

// ---- 4. Nested composites; the outer X has 200 * s elements -----------------
static void nested(std::mt19937_64& rng, std::size_t s) {
    composite_block outer("outer2");
    auto inner_own = std::make_unique<composite_block>("inner");
    composite_block* inner = inner_own.get();
    shared_data_t& od = outer.data();
    shared_data_t& id = inner->data();
    id.set("Z", filled(50, 0.1, 1.0));
    id.declare_dependencies("i1", {"Z", "X", "o1"});
    id.declare_dependencies("i2", {"i1", "X"});
    inner->add_child(std::make_unique<probe_block>("i1", 1, &id));
    inner->add_child(std::make_unique<probe_block>("i2", 2, &id));
    od.set("X", filled(200 * s, 0.7, 0.5));
    od.declare_dependencies("o1", {"X"});
    od.declare_dependencies("inner", {"X", "o1"});
    od.declare_dependencies("o2", {"inner", "X"});
    outer.add_child(std::make_unique<probe_block>("o1", 1, &od));
    outer.add_child(std::move(inner_own));
    outer.add_child(std::make_unique<probe_block>("o2", 1, &od));
    sweeps(outer, rng, 5);
    od.set("X", filled(200 * s, -0.3, 0.9));
    sweeps(outer, rng, 3);
    id.set("Z", filled(80, 0.4, 0.4));
    sweeps(outer, rng, 25);
    od.set("X", filled(200 * s, 0.2, 0.1));
    sweeps(outer, rng, 2);
}

// ---- 5. Two children with the same name -------------------------------------
static void duplicate_names(std::mt19937_64& rng, std::size_t s) {
    composite_block comp("dup");
    shared_data_t& d = comp.data();
    d.set("X", filled(200 * s, 0.2, 0.2));
    d.declare_dependencies("p", {"X", "p"});
    comp.add_child(std::make_unique<probe_block>("p", 1, &d));
    comp.add_child(std::make_unique<probe_block>("p", 1, &d));
    sweeps(comp, rng, 4);
    d.set("X", filled(200 * s, 0.6, 0.1));
    sweeps(comp, rng, 25);
    d.set("X", filled(200 * s, 0.1, 0.9));
    sweeps(comp, rng, 2);
}

// ---- 6. The size rule at its bounds -----------------------------------------
// One probe child t with a single dependency X of n elements, so its context
// takes 8 n bytes. For each n, the child's next 12 updates must come the way
// the rule gives, and readapt_NUTS must pass views.
template <class C>
static void rule_checks(std::mt19937_64& rng) {
    if constexpr (!has_copy_band<C>::value) {
        ok(false, "composite_block::context_copy_min_bytes / context_copy_max_bytes "
                  "are available");
    } else {
        const std::size_t lo = C::context_copy_min_bytes;
        const std::size_t hi = C::context_copy_max_bytes;
        ok(lo == std::size_t(24 * 1024) && hi == std::size_t(160 * 1024),
           "the bounds are those of AI4BAYESCODE_CONTEXT_COPY_MIN_BYTES / "
           "AI4BAYESCODE_CONTEXT_COPY_MAX_BYTES");
        const std::size_t n_lo = (lo + 7) / 8;      // smallest n with 8 n >= lo
        const std::size_t n_hi = hi / 8;            // largest n with 8 n <= hi
        struct step_t { std::size_t n; int way; const char* what; };
        const std::vector<step_t> plan = {
            {1,                 1, "1 element"},
            {n_lo - 1,          1, "just below the band"},
            {n_lo,              0, "at the lower bound"},
            {(n_lo + n_hi) / 2, 0, "inside the band"},
            {n_hi,              0, "at the upper bound"},
            {n_hi + 1,          1, "just above the band"},
            {4 * n_hi,          1, "far above the band"},
            {n_lo,              0, "back at the lower bound"},
            {n_lo - 1,          1, "back below the band"},
            {n_hi + 1,          1, "back above the band"},
            {n_hi,              0, "back at the upper bound"},
        };
        composite_block comp("rule");
        shared_data_t& d = comp.data();
        d.set("X", filled(plan[0].n, 0.3, 0.2));
        d.declare_dependencies("t", {"X"});
        auto own = std::make_unique<probe_block>("t", 1, &d, /*readapt=*/true);
        probe_block* t = own.get();
        comp.add_child(std::move(own));
        for (const auto& st : plan) {
            d.set("X", filled(st.n, 0.1 + 1e-4 * double(st.n % 97), 0.2));
            const std::size_t first = t->modes_.size();
            sweeps(comp, rng, 12);
            bool all = t->modes_.size() == first + 12;
            for (std::size_t k = first; k < t->modes_.size(); ++k)
                if (t->modes_[k] != st.way) all = false;
            readapt(comp, rng, 2);
            const bool views = t->modes_.size() == first + 13 && t->modes_.back() == 1;
            ok(all && views, std::string("rule, ") + st.what + " (" + std::to_string(8 * st.n)
                    + " bytes): every update gets " + (st.way ? "views" : "a copy")
                    + ", readapt_NUTS views");
        }

        // Two values each below the band whose sum is inside it.
        composite_block two("rule2");
        shared_data_t& d2 = two.data();
        d2.set("X", filled(n_lo / 2 + 1, 0.2, 0.3));
        d2.set("y", filled(n_lo / 2 + 1, 0.4, 0.1));
        d2.declare_dependencies("u", {"X", "y"});
        auto own2 = std::make_unique<probe_block>("u", 1, &d2);
        probe_block* u = own2.get();
        two.add_child(std::move(own2));
        sweeps(two, rng, 10);
        bool all = u->modes_.size() == 10;
        for (int m : u->modes_) if (m != 0) all = false;
        ok(all, "rule: the size is the sum over the context's values");

        // A key listed twice counts once, also after its length changes
        // (8 n_hi bytes are inside the band, 16 n_hi bytes above it).
        composite_block dup("rule3");
        shared_data_t& d3 = dup.data();
        d3.set("X", filled(n_lo, 0.1, 0.2));
        d3.declare_dependencies("v", {"X", "X"});
        auto own3 = std::make_unique<probe_block>("v", 1, &d3);
        probe_block* v = own3.get();
        dup.add_child(std::move(own3));
        sweeps(dup, rng, 3);
        d3.set("X", filled(n_hi, 0.3, 0.1));
        sweeps(dup, rng, 3);
        all = v->modes_.size() == 6;
        for (int m : v->modes_) if (m != 0) all = false;
        ok(all, "rule: a key listed twice counts once, also after its length changes");
    }
}

// ---- 7. Direct checks of shared_data_t::context_view_for --------------------

template <class T, class = void>
struct has_context_view : std::false_type {};
template <class T>
struct has_context_view<T, std::void_t<decltype(
    std::declval<T&>().context_view_for(std::declval<const std::string&>()))>>
    : std::true_type {};

template <class T, class = void>
struct has_view_or_copy : std::false_type {};
template <class T>
struct has_view_or_copy<T, std::void_t<decltype(
    std::declval<T&>().context_view_or_copy_for(
        std::declval<const std::string&>(), std::size_t(0), std::size_t(0),
        std::declval<block_context&>()))>>
    : std::true_type {};

// Views of d for `block` equal build_context_for and refer to d's storage.
// context_view_or_copy_for gives a copy equal to build_context_for when the
// size is within the bounds (both bounds included), and the views otherwise.
// The first call after the change is a copy into a map that holds other
// entries, so the copy is made before the views are made again.
template <class SD>
static void check_views(SD& d, const std::string& block, const std::string& what) {
    const block_context fresh = d.build_context_for(block);
    if constexpr (has_view_or_copy<SD>::value) {
        block_context c0;
        for (int k = 0; k < 40; ++k) c0["other" + std::to_string(k)] = arma::vec{double(k)};
        if (!fresh.empty()) c0[fresh.begin()->first] = arma::vec{-1.0, -2.0};
        const block_context& r0 =
            d.context_view_or_copy_for(block, 0, std::size_t(-1), c0);
        bool copied = &r0 == &c0 && same_context(c0, fresh);
        for (const auto& kv : c0)
            if (copied && kv.second.n_elem > 0 &&
                kv.second.memptr() == d.get(kv.first).memptr())
                copied = false;
        ok(copied, what + ": a copy made first, into a map that held other entries, "
                          "equals build_context_for");
    }
    const block_context& v = d.context_view_for(block);
    ok(same_context(v, fresh), what + ": views equal build_context_for");
    bool stored = true;
    for (const auto& kv : v)
        if (kv.second.memptr() != d.get(kv.first).memptr()) stored = false;
    ok(stored, what + ": every entry refers to the stored value (no copy)");
    if constexpr (has_view_or_copy<SD>::value) {
        const std::size_t b = context_bytes(fresh);
        block_context c1, c2, c3, c4;
        c2["other"] = arma::vec{1.0, 2.0};
        const block_context& r1 = d.context_view_or_copy_for(block, b, b, c1);
        bool copied = &r1 == &c1;
        for (const auto& kv : c1)
            if (kv.second.n_elem > 0 && kv.second.memptr() == d.get(kv.first).memptr())
                copied = false;
        ok(copied && same_context(c1, fresh),
           what + ": a size on both bounds gives a copy equal to build_context_for");
        const block_context& r2 = d.context_view_or_copy_for(block, b + 1, b + 100, c2);
        const block_context& r3 =
            d.context_view_or_copy_for(block, 0, b == 0 ? 0 : b - 1, c3);
        const block_context& r4 = d.context_view_or_copy_for(block, 1, 0, c4);
        const block_context& vv = d.context_view_for(block);
        ok(&r2 == &vv && c2.size() == 1 && c2.count("other") &&
           (b == 0 || (&r3 == &vv && c3.empty())) && &r4 == &vv && c4.empty(),
           what + ": a size outside the bounds gives the views and leaves `copy` "
                  "as it was");
    }
}

template <class SD>
static void view_checks() {
    if constexpr (!has_view_or_copy<SD>::value) {
        ok(false, "shared_data_t::context_view_or_copy_for is available");
    }
    if constexpr (!has_context_view<SD>::value) {
        ok(false, "shared_data_t::context_view_for is available");
    } else {
        std::vector<double> ext(50, 1.0);  // caller memory aliased below
        SD d;
        d.set("X", filled(200, 0.5, 1.0));
        d.set("y", filled(30, -0.2, 0.7));
        d.set("c", arma::vec{2.0});
        d.set("e", arma::vec());
        d.set("A", arma::vec(ext.data(), ext.size(), /*copy_aux_mem=*/false,
                             /*strict=*/true));
        d.set("r", arma::vec(30, arma::fill::zeros));
        d.declare_dependencies("b", {"X", "y", "c", "e", "A", "r"});
        d.register_refresher("r", [](const shared_data_t& s) -> arma::vec {
            return s.get("y") * s.get("c")[0];
        });
        d.declare_invalidates("b", {"r"});
        d.register_stochastic_refresher("c",
            [](const shared_data_t&, std::mt19937_64& g) -> arma::vec {
                std::normal_distribution<double> nd(0.0, 1.0);
                return arma::vec{nd(g)};
            });
        check_views(d, "b", "initial");

        // Unchanged data: the same map and the same view objects.
        const block_context& v1 = d.context_view_for("b");
        const arma::vec* x1 = &v1.at("X");
        const block_context& v2 = d.context_view_for("b");
        ok(&v1 == &v2 && &v2.at("X") == x1,
           "repeated call returns the same views, not new ones");

        d.set("X", filled(200, 1.5, 0.3));
        check_views(d, "b", "set same size");
        d.set("X", filled(500, -1.0, 0.2));
        check_views(d, "b", "set new size");
        d.set("c", arma::vec{3.0});
        check_views(d, "b", "set small value");
        d.set("c", arma::vec{3.0, 1.0, 2.0});           // same local buffer
        check_views(d, "b", "small value, new length, same storage");
        d.write_back("c", arma::vec{4.0});
        check_views(d, "b", "write_back");
        const double* x_mem = d.get("X").memptr();
        d.write_back("X", filled(300, 0.7, 0.1));       // shorter, same buffer
        check_views(d, "b", "write_back of a shorter value");
        d.write_back("X", filled(450, 0.2, 0.3));       // longer, same buffer
        check_views(d, "b", "write_back of a longer value within the buffer");
        ok(d.get("X").memptr() == x_mem,
           "the storage address is the same after the two write_backs above");
        d.write_back("y", filled(30, 0.9, 0.1));
        check_views(d, "b", "write_back of a longer value");
        d.refresh_derived_for("b");
        check_views(d, "b", "refresh_derived_for");
        d.set("y", filled(30, 0.2, 0.2));
        d.refresh_key("r");
        check_views(d, "b", "refresh_key");
        d.set("y", filled(30, 0.3, 0.6));
        d.refresh_all();
        check_views(d, "b", "refresh_all");
        {
            std::mt19937_64 g(3);
            d.run_stochastic_refresher("c", g);
        }
        check_views(d, "b", "run_stochastic_refresher");
        arma::vec& Xref = d.get_mut("X");
        check_views(d, "b", "get_mut");
        Xref[0] += 1.0;
        check_views(d, "b", "write through a get_mut reference");
        d.get_mut("y").resize(10);
        check_views(d, "b", "get_mut shrink");
        d.get_mut("y").resize(30);
        check_views(d, "b", "get_mut grow back");
        d.get_mut("y").resize(300);
        check_views(d, "b", "get_mut grow");
        ext[0] = 42.0;  // caller memory written, no call on d
        check_views(d, "b", "write to aliased caller memory");
        ok(d.context_view_for("b").at("A")[0] == 42.0,
           "view of an aliased value shows the caller's write");

        d.declare_dependencies("b", {"y", "X"});
        check_views(d, "b", "dependencies dropped and reordered");
        d.declare_dependencies("b", {"X", "y", "X"});
        check_views(d, "b", "duplicate dependency");
        d.declare_dependencies("b", {});
        check_views(d, "b", "empty dependency list");
        check_views(d, "nobody", "block without dependencies");
        d.declare_dependencies("nobody", {"X", "y"});
        check_views(d, "nobody", "dependencies declared after the first call");
        d.declare_dependencies("b", {"X", "c", "y", "e", "A", "r"});
        check_views(d, "b", "dependencies restored");

        d.declare_dependencies("m", {"X", "nope"});
        std::string expected_msg, got_msg;
        try { (void)d.build_context_for("m"); }
        catch (const std::out_of_range& ex) { expected_msg = ex.what(); }
        try { (void)d.context_view_for("m"); }
        catch (const std::out_of_range& ex) { got_msg = ex.what(); }
        ok(!expected_msg.empty() && got_msg == expected_msg,
           "missing key throws the build_context_for error");
        d.set("nope", arma::vec{1.0});
        check_views(d, "m", "missing key added");

        // Copies and moves: views always refer to the object they come from.
        SD d2(d);
        check_views(d2, "b", "copy-constructed object");
        check_views(d, "b", "source of the copy");
        d2.set("X", filled(500, 0.1, 0.1));
        check_views(d2, "b", "copy after a set");
        check_views(d, "b", "source after a set on the copy");
        SD d3(std::move(d2));
        check_views(d3, "b", "move-constructed object");
        d3 = d;
        check_views(d3, "b", "copy-assigned object");
        d.set("X", filled(200, 0.8, 0.8));
        d3 = std::move(d);
        check_views(d3, "b", "move-assigned object");
        SD& self = d3;
        d3 = self;
        check_views(d3, "b", "self-assigned object");
    }
}

// ---- 8. A stored value that aliases caller memory ---------------------------
static void aliased(std::mt19937_64& rng, std::size_t n) {
    std::vector<double> buf(n, 1.0);
    composite_block comp("alias");
    comp.data().set("A", arma::vec(buf.data(), buf.size(),
                                   /*copy_aux_mem=*/false, /*strict=*/true));
    comp.data().declare_dependencies("r", {"A"});
    auto own = std::make_unique<length_block>("r");
    length_block* r = own.get();
    comp.add_child(std::move(own));
    bool all = true;
    for (int s = 0; s < 30; ++s) {
        buf[0] = double(s);  // written without any call on the composite
        comp.step(rng);
        if (r->first_ != double(s)) all = false;
    }
    ok(all, "every update shows the caller's latest write to aliased memory ("
            + std::to_string(n) + " elements)");
}

// ---- 9. Data-sized allocations in steady sweeps -----------------------------
// A child that copies nothing, with one dependency X of n elements: the
// number of allocations of at least 8 n bytes in 50 sweeps after 5 sweeps.
static std::size_t steady_allocs(std::mt19937_64& rng, std::size_t n) {
    composite_block comp("big");
    comp.data().set("X", filled(n, 0.3, 0.1));
    comp.data().declare_dependencies("s", {"X"});
    comp.add_child(std::make_unique<length_block>("s"));
    sweeps(comp, rng, 5);
    g_big_threshold = n * sizeof(double);
    g_big_allocs = 0;
    sweeps(comp, rng, 50);
    const std::size_t steady = g_big_allocs;
    g_big_threshold = 0;
    g_n_big_live = 0;
    return steady;
}

// ---- 10. Lifetime of a copy -------------------------------------------------
// Child that records, when its context is a copy, whether the copy's buffer
// for X is still allocated in its step() and in the refresh that follows its
// write-back. It stores the buffer's address only to look it up among the
// allocated buffers, and never dereferences it after set_context.
class lifetime_block : public block_sampler {
public:
    lifetime_block(std::string name, const shared_data_t* watched)
        : name_(std::move(name)), cur_(1, arma::fill::zeros), watched_(watched) {}
    void set_context(const block_context& ctx) override {
        mem_ = ctx.at("X").memptr();
        copy_ = mem_ != watched_->get("X").memptr();
        first_ = ctx.at("X")[0];
    }
    void step(std::mt19937_64& rng) override {
        if (copy_) {
            ++n_copy_;
            if (!big_live(mem_)) ++n_freed_in_step_;
        }
        std::normal_distribution<double> nd(0.0, 1.0);
        cur_[0] = first_ + nd(rng);
    }
    const arma::vec& current() const override { return cur_; }
    void set_current(const arma::vec& t) override { cur_ = t; }
    std::size_t dim() const noexcept override { return 1; }
    const std::string& name() const noexcept override { return name_; }

    const double* mem_ = nullptr;
    bool copy_ = false;
    int n_copy_ = 0, n_freed_in_step_ = 0, n_freed_in_refresh_ = 0;

private:
    std::string name_;
    arma::vec cur_;
    const shared_data_t* watched_;
    double first_ = 0.0;
};

static void copy_lifetime(std::mt19937_64& rng, std::size_t n) {
    composite_block comp("life");
    shared_data_t& d = comp.data();
    d.set("X", filled(n, 0.2, 0.4));
    d.set("lr", arma::vec{0.0});
    d.declare_dependencies("lt", {"X"});
    auto own = std::make_unique<lifetime_block>("lt", &d);
    lifetime_block* lt = own.get();
    comp.add_child(std::move(own));
    d.register_refresher("lr", [lt](const shared_data_t& sd) -> arma::vec {
        if (lt->copy_ && !big_live(lt->mem_)) ++lt->n_freed_in_refresh_;
        return arma::vec{sd.get("lt")[0]};
    });
    d.declare_invalidates("lt", {"lr"});
    g_big_threshold = n * sizeof(double);
    g_n_big_live = 0;
    int n_after = 0, n_freed_after = 0;
    for (int s = 0; s < 20; ++s) {
        comp.step(rng);
        if (lt->copy_) {
            ++n_after;
            if (!big_live(lt->mem_)) ++n_freed_after;
        }
    }
    g_big_threshold = 0;
    g_n_big_live = 0;
    ok(lt->n_copy_ == 20 && n_after == 20,
       "lifetime: the child got a copy in " + std::to_string(lt->n_copy_)
       + " of 20 updates (" + std::to_string(n) + " elements)");
    ok(lt->n_freed_in_step_ == 0,
       "lifetime: the copy is still allocated in the child's step()");
    ok(lt->n_freed_in_refresh_ == 0,
       "lifetime: the copy is still allocated in the refresh after the write-back");
    ok(n_freed_after == n_after,
       "lifetime: the copy is freed once the child's update is done");
}

// ---- 12. A dependency list declared after the child's first update ---------
// The probe child t is swept with no dependency list (an empty context), then
// given the list {X}, with X of n elements, and swept again; every context
// is checked against build_context_for and the size rule.
static void late_list(std::mt19937_64& rng, std::size_t n) {
    composite_block comp("late");
    shared_data_t& d = comp.data();
    d.set("X", filled(n, 0.4, 0.3));
    auto own = std::make_unique<probe_block>("t", 1, &d);
    probe_block* t = own.get();
    comp.add_child(std::move(own));
    sweeps(comp, rng, 3);
    d.declare_dependencies("t", {"X"});
    sweeps(comp, rng, 3);
    const int way = region_of(8 * n) == 1 ? 0 : 1;
    bool all = t->n_set_ == 6 && t->modes_.size() == 6;
    for (std::size_t k = 0; all && k < 6; ++k)
        if (t->modes_[k] != (k < 3 ? -1 : way)) all = false;
    ok(all, "dependency list declared after 3 updates (" + std::to_string(8 * n)
            + " bytes): empty contexts, then " + (way ? "views" : "copies") + " of X");
}

// ---- 13. The views are not made again for a copy --------------------------
// Making the views again allocates a node per key of the context and the
// buckets; for a new dependency list it also allocates the key list and the
// pointer vector. A copy allocates a node per key and the buckets. So, for a
// copy made after a value got new storage or a new length,
// context_view_or_copy_for() calls the global operator new as often as for a
// copy of unchanged values, unless it makes the views again; and views made
// again after a value got new storage take as many calls as a copy.
template <class SD>
static std::size_t news_of_copy(SD& d, const std::string& block) {
    block_context c;
    g_news = 0;
    g_count_news = true;
    (void)d.context_view_or_copy_for(block, 0, std::size_t(-1), c);
    g_count_news = false;
    return g_news;
}

// Steady sweeps of children a (context {s}) and b (context {X, E}, 40008
// bytes, inside the band); E is refreshed after a, into a new vector, when
// `refresh` is set. The number of calls of operator new in 20 sweeps.
template <class SD>
static std::size_t news_of_sweeps(std::mt19937_64& rng, bool refresh) {
    composite_block comp("renew");
    SD& d = comp.data();
    d.set("s", arma::vec{0.5});
    d.set("X", filled(4961, 0.3, 0.2));
    d.set("E", filled(40, 0.1, 0.1));
    d.declare_dependencies("a", {"s"});
    d.declare_dependencies("b", {"X", "E"});
    d.register_refresher("E", [](const shared_data_t& sd) -> arma::vec {
        return filled(40, sd.get("a")[0], 0.1);
    });
    if (refresh) d.declare_invalidates("a", {"E"});
    comp.add_child(std::make_unique<length_block>("a"));
    comp.add_child(std::make_unique<length_block>("b"));
    sweeps(comp, rng, 5);
    const double* e_mem = d.get("E").memptr();
    sweeps(comp, rng, 1);
    ok((d.get("E").memptr() != e_mem) == refresh,
       std::string("E ") + (refresh ? "gets" : "does not get") + " new storage in a sweep");
    g_news = 0;
    g_count_news = true;
    sweeps(comp, rng, 20);
    g_count_news = false;
    const std::size_t n = g_news;
    check_views(d, "b", std::string("after steady sweeps, E ")
                        + (refresh ? "refreshed" : "unchanged"));
    return n;
}

template <class SD>
static void copy_without_views(std::mt19937_64& rng) {
    if constexpr (has_view_or_copy<SD>::value) {
        SD d;
        d.set("X", filled(3000, 0.2, 0.3));
        d.set("E", filled(40, 0.1, 0.5));
        d.declare_dependencies("w", {"X", "E", "X"});
        (void)news_of_copy(d, "w");                    // makes the views entry
        const std::size_t same = news_of_copy(d, "w");
        const double* e_mem = d.get("E").memptr();
        d.set("E", filled(40, 0.7, 0.2));             // new storage
        const bool moved = d.get("E").memptr() != e_mem;
        const std::size_t new_storage = news_of_copy(d, "w");
        d.set("E", filled(60, 0.7, 0.2));             // new storage and length
        const std::size_t new_length = news_of_copy(d, "w");
        std::printf("  calls of operator new for a copy: unchanged values %zu, a value "
                    "with new storage %zu, with a new length %zu\n",
                    same, new_storage, new_length);
        ok(same > 0 && moved && new_storage == same && new_length == same,
           "the views are not made again for a copy after a value got new storage or "
           "a new length");
        check_views(d, "w", "views after copies made while E had new storage");
        d.declare_dependencies("w", {"X", "E"});
        const std::size_t new_list = news_of_copy(d, "w");
        ok(new_list > same,
           "the views are made again for a copy after the dependency list changed ("
           + std::to_string(new_list) + " calls of operator new)");
        check_views(d, "w", "views after the dependency list changed");

        d.declare_dependencies("u", {"X", "E"});
        (void)news_of_copy(d, "u");                    // makes the views entry
        const std::size_t copy_u = news_of_copy(d, "u");
        (void)d.context_view_for("u");
        d.set("E", filled(60, 0.3, 0.3));             // new storage, same length
        g_news = 0;
        g_count_news = true;
        (void)d.context_view_for("u");
        g_count_news = false;
        const std::size_t remade = g_news;
        ok(copy_u > 0 && remade == copy_u,
           "views made again after a value got new storage take as many calls of "
           "operator new as a copy (" + std::to_string(remade) + " and "
           + std::to_string(copy_u)
           + "): the key list and the pointers are not made again");
        check_views(d, "u", "views made again after a value got new storage");
        d.set("E", filled(60, 0.5, 0.4));             // new storage, same length
        {
            block_context cu;
            g_news = 0;
            g_count_news = true;
            const block_context& vu = d.context_view_or_copy_for("u", 1, 0, cu);
            g_count_news = false;
            ok(&vu != &cu && g_news == copy_u,
               "views returned by context_view_or_copy_for, made again after a value got "
               "new storage, take as many calls of operator new as a copy ("
               + std::to_string(g_news) + " and " + std::to_string(copy_u) + ")");
        }
        check_views(d, "u", "views of context_view_or_copy_for made again after new storage");

        const std::size_t n_refresh = news_of_sweeps<SD>(rng, true);
        const std::size_t n_fixed = news_of_sweeps<SD>(rng, false);
        std::printf("  calls of operator new in 20 steady sweeps: E refreshed into a "
                    "new vector %zu, E unchanged %zu\n", n_refresh, n_fixed);
        ok(n_refresh == n_fixed,
           "steady sweeps call operator new as often with E refreshed into a new "
           "vector as with E unchanged");
    }
}

// ---- 11. sync_child ---------------------------------------------------------

template <class C, class = void>
struct has_sync_child : std::false_type {};
template <class C>
struct has_sync_child<C, std::void_t<decltype(
    std::declval<C&>().sync_child(std::size_t(0)))>> : std::true_type {};

// Child with two named outputs, u and w, and a loader of its own.
class two_out_block : public block_sampler {
public:
    explicit two_out_block(std::string name)
        : name_(std::move(name)), cur_{0.5, -0.5} {}
    void set_context(const block_context& ctx) override {
        k_ = ctx.count("X") ? ctx.at("X")[0] : 0.0;
    }
    void step(std::mt19937_64& rng) override {
        std::normal_distribution<double> nd(0.0, 1.0);
        cur_[0] = 0.5 * cur_[0] + 0.1 * k_ + nd(rng);
        cur_[1] = 0.5 * cur_[1] - 0.1 * k_ + nd(rng);
    }
    const arma::vec& current() const override { return cur_; }
    void set_current(const arma::vec& t) override { cur_ = t; ++n_set_current_; }
    std::size_t dim() const noexcept override { return 2; }
    const std::string& name() const noexcept override { return name_; }
    state_map current_named_outputs() const override {
        return {{"u", arma::vec{cur_[0]}}, {"w", arma::vec{cur_[1]}}};
    }
    void load(double u, double w) { cur_[0] = u; cur_[1] = w; }
    int n_set_current_ = 0;

private:
    std::string name_;
    arma::vec cur_;
    double k_ = 0.0;
};

// Child whose named output differs between the overloads with and without an
// RNG, as a VI block's does (a q-draw with an RNG, the q-mean without).
class two_overload_block : public block_sampler {
public:
    explicit two_overload_block(std::string name)
        : name_(std::move(name)), cur_{1.0} {}
    void set_context(const block_context&) override {}
    void step(std::mt19937_64& rng) override {
        std::normal_distribution<double> nd(0.0, 1.0);
        cur_[0] = 0.9 * cur_[0] + 0.1 * nd(rng);
    }
    const arma::vec& current() const override { return cur_; }
    void set_current(const arma::vec& t) override { cur_ = t; }
    std::size_t dim() const noexcept override { return 1; }
    const std::string& name() const noexcept override { return name_; }
    state_map current_named_outputs(std::mt19937_64& rng) const override {
        std::normal_distribution<double> nd(0.0, 1.0);
        return {{name_, arma::vec{cur_[0] + 100.0 + nd(rng)}}};
    }
    void load(double m) { cur_[0] = m; }

private:
    std::string name_;
    arma::vec cur_;
};

// The helpers below are [[maybe_unused]]: the checks that call them compile
// only on trees that have sync_child.

// Probes pa, pb and pc on X, y, c with derived r (by pa), plus the
// two-output child "two" with derived uw = u + w, and the child "vi".
[[maybe_unused]] static void build_sync_model(composite_block& comp) {
    build_model(comp, 1);
    shared_data_t& d = comp.data();
    d.set("uw", arma::vec{0.0});
    d.declare_dependencies("two", {"X", "pa"});
    d.register_refresher("uw", [](const shared_data_t& sd) -> arma::vec {
        return arma::vec{sd.get("u")[0] + sd.get("w")[0]};
    });
    d.declare_invalidates("two", {"uw"});
    comp.add_child(std::make_unique<two_out_block>("two"));
    comp.add_child(std::make_unique<two_overload_block>("vi"));
    d.refresh_all();
}

[[maybe_unused]] static bool same_values(const shared_data_t& a, const shared_data_t& b) {
    if (a.values().size() != b.values().size()) return false;
    for (const auto& kv : a.values()) {
        if (!b.has(kv.first)) return false;
        const arma::vec& y = b.get(kv.first);
        if (kv.second.n_elem != y.n_elem) return false;
        if (kv.second.n_elem > 0 &&
            std::memcmp(kv.second.memptr(), y.memptr(),
                        kv.second.n_elem * sizeof(double)) != 0)
            return false;
    }
    return true;
}

[[maybe_unused]] static bool same_vec(const arma::vec& a, const arma::vec& b) {
    return a.n_elem == b.n_elem &&
           (a.n_elem == 0 ||
            std::memcmp(a.memptr(), b.memptr(), a.n_elem * sizeof(double)) == 0);
}

template <class C>
static void sync_child_checks() {
    if constexpr (!has_sync_child<C>::value) {
        ok(false, "composite_block::sync_child is available");
    } else {
        // Twins: A is restored with set_current(theta), B with a loader on
        // each child followed by sync_child.
        C A("twin"), B("twin");
        build_sync_model(A);
        build_sync_model(B);
        std::mt19937_64 ra(77), rb(77);
        sweeps(A, ra, 4);
        sweeps(B, rb, 4);
        ok(same_values(A.data(), B.data()), "sync_child: twins agree before the restore");
        const arma::vec theta{0.3, -0.7, 0.2, 0.9, 1.5, -2.5, 0.25};
        ok(theta.n_elem == A.dim(), "sync_child: theta has the composite's dim");
        (void)A.current();
        (void)B.current();
        A.set_current(theta);

        auto& pa = dynamic_cast<probe_block&>(B.child(0));
        const int pa_calls = pa.n_set_current_;
        const arma::vec pa_old = B.data().get("pa");
        const arma::vec r_old = B.data().get("r");
        pa.set_current(theta.subvec(0, 0));
        ok(same_vec(B.data().get("pa"), pa_old) && same_vec(B.data().get("r"), r_old),
           "sync_child: a loader on the child alone leaves the shared data as it was");
        B.sync_child(0);
        ok(pa.n_set_current_ == pa_calls + 1,
           "sync_child: the child's set_current is not called again");
        ok(B.data().get("pa")[0] == theta[0],
           "sync_child: the child's output is in the shared data");
        ok(same_vec(B.data().get("r"), B.data().get("y") * theta[0]),
           "sync_child: the derived key the child invalidates is refreshed");
        B.child(1).set_current(theta.subvec(1, 2));
        B.sync_child(1);
        B.child(2).set_current(theta.subvec(3, 3));
        B.sync_child(2);
        auto& two = dynamic_cast<two_out_block&>(B.child(3));
        const int two_calls = two.n_set_current_;
        two.load(theta[4], theta[5]);                // the child's own loader
        B.sync_child(3);
        ok(two.n_set_current_ == two_calls, "sync_child: no call to set_current");
        ok(B.data().get("u")[0] == theta[4] && B.data().get("w")[0] == theta[5],
           "sync_child: every named output of a multi-output child is written");
        ok(B.data().get("uw")[0] == theta[4] + theta[5],
           "sync_child: derived key of a multi-output child is refreshed");
        dynamic_cast<two_overload_block&>(B.child(4)).load(theta[6]);
        B.sync_child(4);
        ok(B.data().get("vi")[0] == theta[6],
           "sync_child: the output comes from the overload without an RNG");

        ok(same_values(A.data(), B.data()),
           "sync_child: shared data equals that of set_current(theta)");
        ok(same_vec(A.current(), B.current()) && same_vec(B.current(), theta),
           "sync_child: current() is rebuilt and equals that of set_current(theta)");
        sweeps(A, ra, 6);
        sweeps(B, rb, 6);
        ok(same_values(A.data(), B.data()) && same_vec(A.current(), B.current()),
           "sync_child: the sweeps after the restore give the same draws");

        bool threw = false;
        try { B.sync_child(B.n_children()); }
        catch (const std::out_of_range&) { threw = true; }
        ok(threw, "sync_child: an index past the last child throws std::out_of_range");
    }
}

int main() {
    std::mt19937_64 rng(20261122);
    // Used by the sections that run on trees with the size rule only, so the
    // draws of the other sections are the same on every tree.
    std::mt19937_64 rng_rule(99);

    // ---- 1-5 on both ways ------------------------------------------------
    // With s = 1 every context of pa, pb, pd, o1, inner, o2, i1, i2 and p is
    // below the copy band (views); with s = 30 X takes 6000 to 15000
    // elements, which puts those contexts inside the band (copies).
    for (std::size_t s : {std::size_t(1), std::size_t(30)}) {
        write_paths(rng, s);
        get_mut_writes(rng, s);
        moved_in(rng, s);
        nested(rng, s);
        duplicate_names(rng, s);
    }
    if (k_has_rule) {
        ok(g_ways[0][1] > 0 && g_ways[1][0] > 0,
           "sections 1-5 ran with views below the band and copies inside it");
        ok(g_readapt_ways[1] > 0 && g_readapt_ways[0] == 0,
           "readapt_NUTS passed views, also with a context inside the band");
    } else {
        ok(false, "composite_block::context_copy_min_bytes / context_copy_max_bytes "
                  "are available");
    }

    // ---- 8. A stored value that aliases caller memory (views) ------------
    aliased(rng, 100);

    // ---- 6. The size rule at its bounds (trees with the rule only) -------
    g_hashing = false;
    rule_checks<composite_block>(rng_rule);
    if (k_has_rule) {
        // ---- 12. A dependency list declared after the first update ------
        const std::size_t n_lo = (band_min<composite_block>() + 7) / 8;
        const std::size_t n_hi = band_max<composite_block>() / 8;
        late_list(rng_rule, n_lo - 1);
        late_list(rng_rule, n_lo);
        late_list(rng_rule, n_hi + 1);
        ok(g_ways[0][1] > 0 && g_ways[1][0] > 0 && g_ways[2][1] > 0,
           "both ways reached: views below and above the band, copies inside");
        ok(g_ways[0][0] == 0 && g_ways[1][1] == 0 && g_ways[2][0] == 0,
           "no context came the other way");
        std::printf("  contexts passed by step(): below the band %d views, %d copies; "
                    "inside %d copies, %d views; above %d views, %d copies; "
                    "by readapt_NUTS: %d views, %d copies\n",
                    g_ways[0][1], g_ways[0][0], g_ways[1][0], g_ways[1][1],
                    g_ways[2][1], g_ways[2][0], g_readapt_ways[1], g_readapt_ways[0]);
    }
    g_hashing = true;

    // ---- 7. context_view_for, directly -----------------------------------
    view_checks<shared_data_t>();

    // ---- 9. Data-sized allocations in steady sweeps, views ---------------
    {
        const std::size_t n_above = steady_allocs(rng, 1000000);
        std::printf("  data-sized allocations in 50 steady sweeps above the band: %zu\n",
                    n_above);
        ok(n_above == 0, "steady sweeps above the band do not copy unchanged data");
    }

    // ---- 8-10 inside the band, and 9 below it (trees with the rule only) -
    if (k_has_rule) {
        g_hashing = false;
        const std::size_t lo_n = band_min<composite_block>() / 8;
        const std::size_t mid =
            (band_min<composite_block>() + band_max<composite_block>()) / 16;
        aliased(rng_rule, mid);                                   // 8, copies
        copy_lifetime(rng_rule, mid);                             // 10
        const std::size_t n_inside = steady_allocs(rng_rule, mid);
        std::printf("  data-sized allocations in 50 steady sweeps inside the band: %zu\n",
                    n_inside);
        ok(n_inside == 50, "steady sweeps inside the band make one copy each");
        const std::size_t n_below = steady_allocs(rng_rule, lo_n / 2);
        std::printf("  data-sized allocations in 50 steady sweeps below the band: %zu\n",
                    n_below);
        ok(n_below == 0, "steady sweeps below the band do not copy unchanged data");
        // ---- 13. The views are not made again for a copy ----------------
        copy_without_views<shared_data_t>(rng_rule);
        g_hashing = true;
    }

    // ---- 11. sync_child --------------------------------------------------
    g_hashing = false;
    sync_child_checks<composite_block>();
    g_hashing = true;

    std::printf("test_fix_composite_context_cache: %d checks, %d failures, "
                "hash of the probe children's contexts and draws in sections 1-5 "
                "= %016llx\n",
                g_checks, g_fails, (unsigned long long)g_hash);
    if (g_fails == 0) {
        std::printf("PASS\n");
        return 0;
    }
    std::printf("FAIL\n");
    return 1;
}

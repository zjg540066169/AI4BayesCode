// Regression test for the warning composite_block::freeze returns when the
// frozen block invalidates derived keys, and for
// shared_data_t::declare_refreshed_after_set_current.
//
// freeze(name) on a block B whose declare_invalidates list is K returns
//   "freezing 'B' will hold derived key(s) [k1, k2, ...] stale under
//    subsequent set_current updates to upstream data; downstream siblings
//    may sample on unrefreshed values"
// listing the keys of K, in declaration order, that were not declared with
// declare_refreshed_after_set_current. No such warning is returned when every
// key of K is declared, for quiet = true, or for a sub-name freeze. Checked:
//  - without any declaration the text lists every key of K (the text of
//    trees without the declaration);
//  - a sub-name freeze of a block whose freeze returns the warning, by the
//    bare sub-name and by "<block>.<sub-name>", returns none;
//  - declared keys are left out, repeated declarations add to the set, and
//    no warning remains once all keys are declared;
//  - keys declared for other blocks only, or not derived at all, change
//    nothing;
//  - the "already frozen" warning of a second freeze is unchanged;
//  - copy construction, copy assignment and move of shared_data_t carry the
//    declaration, and composite_block::data() = another object replaces it;
//  - a nested composite uses its own declarations for "inner.B";
//  - the accessor returns the declared keys.
// declare_refreshed_after_set_current is detected at compile time, so the
// test also builds against trees without it, and then fails.

#include <armadillo>
#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/shared_data.hpp"
#include "AI4BayesCode/composite_block.hpp"

#include <algorithm>
#include <cstdio>
#include <memory>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using AI4BayesCode::block_context;
using AI4BayesCode::block_sampler;
using AI4BayesCode::composite_block;
using AI4BayesCode::shared_data_t;

static int g_checks = 0, g_fails = 0;

static void ok(bool cond, const std::string& what) {
    ++g_checks;
    if (!cond) { ++g_fails; std::printf("  [FAIL] %s\n", what.c_str()); }
}

template <class SD, class = void>
struct has_declaration : std::false_type {};
template <class SD>
struct has_declaration<SD, std::void_t<
    decltype(std::declval<SD&>().declare_refreshed_after_set_current(
        std::vector<std::string>{})),
    decltype(std::declval<const SD&>().refreshed_after_set_current())>>
    : std::true_type {};

class scalar_block : public block_sampler {
public:
    explicit scalar_block(std::string name)
        : name_(std::move(name)), cur_(1, arma::fill::zeros) {}
    void set_context(const block_context&) override {}
    void step(std::mt19937_64& rng) override {
        std::normal_distribution<double> nd(0.0, 1.0);
        cur_[0] = nd(rng);
    }
    const arma::vec& current() const override { return cur_; }
    void set_current(const arma::vec& t) override { cur_ = t; }
    std::size_t dim() const noexcept override { return 1; }
    const std::string& name() const noexcept override { return name_; }

private:
    std::string name_;
    arma::vec cur_;
};

// A scalar_block with sub-names s1 and s2 that can be frozen one at a time.
class sub_block : public scalar_block {
public:
    using scalar_block::scalar_block;
    std::vector<std::string> subnames() const override { return {"s1", "s2"}; }
    void freeze_sub(const std::string& sub) override { frozen_.push_back(sub); }
    void unfreeze_sub(const std::string& sub) override {
        frozen_.erase(std::remove(frozen_.begin(), frozen_.end(), sub), frozen_.end());
    }
    std::vector<std::string> frozen_subnames() const override { return frozen_; }

private:
    std::vector<std::string> frozen_;
};

static std::string stale_text(const std::string& block, const std::string& keys) {
    return "freezing '" + block + "' will hold derived key(s) [" + keys +
           "] stale under subsequent set_current updates to upstream data; "
           "downstream siblings may sample on unrefreshed values";
}

static std::string show(const std::vector<std::string>& w) {
    std::string s = "{";
    for (const auto& x : w) s += (s.size() > 1 ? " | " : "") + x;
    return s + "}";
}

// Children a, b, c; b invalidates k1, k2, k3 (in that order), c invalidates
// k4; refreshers registered for all four keys.
static void build(composite_block& comp) {
    shared_data_t& d = comp.data();
    d.set("y", arma::vec{1.0, 2.0});
    for (const char* k : {"k1", "k2", "k3", "k4"}) d.set(k, arma::vec{0.0});
    d.declare_dependencies("a", {"y"});
    d.declare_dependencies("b", {"y", "a"});
    d.declare_dependencies("c", {"k1", "k4"});
    d.register_refresher("k1", [](const shared_data_t& s) { return arma::vec{s.get("b")[0]}; });
    d.register_refresher("k2", [](const shared_data_t& s) { return arma::vec{2 * s.get("b")[0]}; });
    d.register_refresher("k3", [](const shared_data_t& s) { return s.get("y") * s.get("b")[0]; });
    d.register_refresher("k4", [](const shared_data_t& s) { return arma::vec{s.get("c")[0]}; });
    d.declare_invalidates("b", {"k1", "k2", "k3"});
    d.declare_invalidates("c", {"k4"});
    comp.add_child(std::make_unique<scalar_block>("a"));
    comp.add_child(std::make_unique<scalar_block>("b"));
    comp.add_child(std::make_unique<scalar_block>("c"));
    d.refresh_all();
}

// The warnings of freeze({name}) on a fresh copy of the composite state:
// freezes, then unfreezes so the next call starts unfrozen.
static std::vector<std::string> freeze_warnings(composite_block& comp,
                                                const std::string& name,
                                                bool quiet = false) {
    auto w = comp.freeze(std::vector<std::string>{name}, quiet);
    comp.unfreeze_all();
    return w;
}

template <class SD>
static void declaration_checks() {
    if constexpr (!has_declaration<SD>::value) {
        ok(false, "shared_data_t::declare_refreshed_after_set_current is available");
    } else {
        // SD is shared_data_t; naming it SD makes these calls dependent, so
        // they are not compiled on trees without the declaration.
        composite_block comp("m");
        build(comp);
        SD& d = comp.data();
        std::mt19937_64 rng(5);
        comp.step(rng);

        ok(d.refreshed_after_set_current().empty(), "nothing declared at first");

        d.declare_refreshed_after_set_current({"k2"});
        auto w = freeze_warnings(comp, "b");
        ok(w == std::vector<std::string>{stale_text("b", "k1, k3")},
           "a declared key is left out: " + show(w));

        d.declare_refreshed_after_set_current({"k4", "not_derived"});
        w = freeze_warnings(comp, "b");
        ok(w == std::vector<std::string>{stale_text("b", "k1, k3")},
           "keys of other blocks and keys that are not derived change nothing: " + show(w));
        ok(freeze_warnings(comp, "c").empty(),
           "every key of c declared: no warning for freeze(\"c\")");

        d.declare_refreshed_after_set_current({"k3"});
        d.declare_refreshed_after_set_current({"k3"});
        w = freeze_warnings(comp, "b");
        ok(w == std::vector<std::string>{stale_text("b", "k1")},
           "repeated declarations add to the set: " + show(w));

        const auto& keys = d.refreshed_after_set_current();
        ok(keys.size() == 4 && keys.count("k2") && keys.count("k3") &&
           keys.count("k4") && keys.count("not_derived"),
           "the accessor returns the declared keys");

        // Copies and moves carry the declaration.
        SD snap = d;
        d.declare_refreshed_after_set_current({"k1"});
        w = freeze_warnings(comp, "b");
        ok(w.empty(), "every key of b declared: no warning: " + show(w));
        w = comp.freeze(std::vector<std::string>{"b"}, false);
        w = comp.freeze(std::vector<std::string>{"b"}, false);
        ok(w == std::vector<std::string>{
                    "block 'b' already frozen; freeze is idempotent"},
           "a second freeze returns only the idempotent warning: " + show(w));
        comp.unfreeze_all();
        comp.data() = snap;                    // without k1
        w = freeze_warnings(comp, "b");
        ok(w == std::vector<std::string>{stale_text("b", "k1")},
           "copy assignment replaces the declaration: " + show(w));
        SD copy(comp.data());
        ok(copy.refreshed_after_set_current() == d.refreshed_after_set_current(),
           "copy construction carries the declaration");
        SD moved(std::move(copy));
        ok(moved.refreshed_after_set_current().size() == 4 &&
           !moved.refreshed_after_set_current().count("k1"),
           "move construction carries the declaration");
        moved.declare_refreshed_after_set_current({"k1"});
        comp.data() = std::move(moved);
        ok(freeze_warnings(comp, "b").empty(), "move assignment carries the declaration");
        comp.step(rng);

        // Nested composite: "inner.b" uses the inner declarations.
        composite_block outer("outer");
        auto inner_own = std::make_unique<composite_block>("inner");
        build(*inner_own);
        SD& id = inner_own->data();
        id.declare_refreshed_after_set_current({"k1", "k3"});
        outer.data().set("z", arma::vec{1.0});
        outer.data().declare_dependencies("inner", {"z"});
        outer.add_child(std::move(inner_own));
        SD& od = outer.data();
        od.declare_refreshed_after_set_current({"k2"});  // outer's own set
        w = outer.freeze(std::vector<std::string>{"inner.b"}, false);
        ok(w == std::vector<std::string>{stale_text("b", "k2")},
           "a nested composite uses its own declarations: " + show(w));
    }
}

// Child j with sub-names s1 and s2 invalidates k5. Freezing j returns the
// warning; freezing s1 or j.s2 returns none, since j's step() and the
// refresh of k5 still run. These hold on every tree.
static void sub_name_checks() {
    composite_block comp("s");
    shared_data_t& d = comp.data();
    d.set("y", arma::vec{1.0, 2.0});
    d.set("k5", arma::vec{0.0});
    d.declare_dependencies("j", {"y"});
    d.register_refresher("k5", [](const shared_data_t& s) { return arma::vec{s.get("j")[0]}; });
    d.declare_invalidates("j", {"k5"});
    comp.add_child(std::make_unique<sub_block>("j"));
    d.refresh_all();
    std::mt19937_64 rng(11);
    comp.step(rng);
    auto w = freeze_warnings(comp, "j");
    ok(w == std::vector<std::string>{stale_text("j", "k5")},
       "freezing the block with sub-names lists its derived key: " + show(w));
    w = comp.freeze(std::vector<std::string>{"s1"}, false);
    const auto fs1 = comp.get_frozen();
    comp.unfreeze_all();
    ok(w.empty() && fs1 == std::vector<std::string>{"j.s1"},
       "freezing the bare sub-name s1 freezes j.s1 and returns no warning: " + show(w));
    w = comp.freeze(std::vector<std::string>{"j.s2"}, false);
    const auto fs2 = comp.get_frozen();
    comp.unfreeze_all();
    ok(w.empty() && fs2 == std::vector<std::string>{"j.s2"},
       "freezing j.s2 freezes it and returns no warning: " + show(w));
}

int main() {
    // Without any declaration: the text lists every key, in declaration
    // order; quiet and sub-name freezes give none. These hold on every tree.
    {
        composite_block comp("m");
        build(comp);
        std::mt19937_64 rng(3);
        comp.step(rng);
        auto w = freeze_warnings(comp, "b");
        ok(w == std::vector<std::string>{stale_text("b", "k1, k2, k3")},
           "no declaration: every derived key of b is listed: " + show(w));
        w = freeze_warnings(comp, "c");
        ok(w == std::vector<std::string>{stale_text("c", "k4")},
           "no declaration: the derived key of c is listed: " + show(w));
        ok(freeze_warnings(comp, "a").empty(), "a block that invalidates nothing: no warning");
        ok(freeze_warnings(comp, "b", /*quiet=*/true).empty(), "quiet = true: no warning");
    }
    sub_name_checks();

    declaration_checks<shared_data_t>();

    std::printf("test_fix_composite_stale_warning: %d checks, %d failures\n",
                g_checks, g_fails);
    if (g_fails == 0) {
        std::printf("PASS\n");
        return 0;
    }
    std::printf("FAIL\n");
    return 1;
}

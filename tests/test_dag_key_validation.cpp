// Regression test for the two declare_invalidates / Gibbs-DAG defects.
//
//   Defect 1 (silent freeze): a declare_invalidates / declare_dependencies
//     key that is not a child BLOCK name is a silent no-op --
//     refresh_derived_for / build_context_for return without acting, the
//     derived key freezes at its initial value, and the sampler converges to
//     the WRONG posterior with no exception, no NaN, healthy R-hat and ESS.
//     Fix: composite_block::validate_dag_keys_ runs on the first step() and
//     throws std::invalid_argument on any declared name that is not a child.
//
//   Defect 2 (assign, not append): declare_invalidates used
//     `invalidates_[name] = keys`, so a second call for the same block
//     silently dropped the first. Fix: it now accumulates (set-union).
//
// Both defects fail SILENTLY in production (wrong posterior, no error), so
// this test asserts the loud behaviour the fixes introduce.
#include <armadillo>
#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/shared_data.hpp"
#include "AI4BayesCode/composite_block.hpp"
#include <algorithm>
#include <cstdio>
#include <memory>
#include <stdexcept>

using AI4BayesCode::block_sampler;
using AI4BayesCode::block_context;
using AI4BayesCode::composite_block;
using AI4BayesCode::shared_data_t;

// Minimal concrete block: holds a name + a scalar; step() holds the value.
struct stub_block : block_sampler {
    std::string nm_;
    arma::vec   val_;
    stub_block(std::string n, double v) : nm_(std::move(n)), val_(arma::vec{v}) {}
    void set_context(const block_context&) override {}
    void step(std::mt19937_64&) override {}                       // hold
    const arma::vec& current() const override { return val_; }
    void set_current(const arma::vec& t) override { val_ = t; }
    std::size_t dim() const noexcept override { return val_.n_elem; }
    const std::string& name() const noexcept override { return nm_; }
};

static int fails = 0, checks = 0;
static void ok(bool cond, const char* what) {
    ++checks; if (!cond) ++fails;
    std::printf("  [%s] %s\n", cond ? "PASS" : "FAIL", what);
}

int main() {
    std::mt19937_64 rng(1);

    // ---- Defect 2: declare_invalidates accumulates (set-union) -----------
    {
        shared_data_t sd;
        sd.declare_invalidates("A", {"D1"});
        sd.declare_invalidates("A", {"D2"});          // must NOT drop D1
        sd.declare_invalidates("A", {"D1", "D3"});    // D1 dedup, D3 added
        const auto& v = sd.invalidates().at("A");
        ok(v.size() == 3, "append: 3 keys retained across 3 calls");
        bool has = std::find(v.begin(), v.end(), "D1") != v.end()
                && std::find(v.begin(), v.end(), "D2") != v.end()
                && std::find(v.begin(), v.end(), "D3") != v.end();
        ok(has, "append: {D1,D2,D3} all present");
        long nd1 = std::count(v.begin(), v.end(), std::string("D1"));
        ok(nd1 == 1, "append: D1 appears exactly once (deduped)");
    }

    // ---- Correct usage: refresh fires on step (control) ------------------
    {
        composite_block comp("root_ok");
        comp.add_child(std::make_unique<stub_block>("A", 3.0));
        comp.add_child(std::make_unique<stub_block>("B", 0.0));
        comp.data().register_refresher("D", [](const shared_data_t& d) {
            return arma::vec{ 2.0 * d.get("A")[0] };
        });
        comp.data().declare_invalidates("A", {"D"});
        comp.data().refresh_all();                          // D = 6 (A=3)
        ok(std::abs(comp.data().get("D")[0] - 6.0) < 1e-12, "prime: D = 2A = 6");
        // bump A via the child so write_back carries the new value
        static_cast<stub_block&>(comp.child(0)).set_current(arma::vec{5.0});
        comp.step(rng);                                     // D must refresh
        ok(std::abs(comp.data().get("D")[0] - 10.0) < 1e-12,
           "step: refresh_derived_for fired -> D = 2*5 = 10");
    }

    // ---- Defect 1a: wrong invalidates key (sub-param) throws on step -----
    {
        composite_block comp("root_bad_inval");
        comp.add_child(std::make_unique<stub_block>("A", 3.0));
        comp.data().register_refresher("D", [](const shared_data_t& d) {
            return arma::vec{ 2.0 * d.get("A")[0] };
        });
        // "beta0" reads as "when beta0 changes refresh D" but is NOT a child.
        comp.data().declare_invalidates("beta0", {"D"});
        comp.data().refresh_all();
        bool threw = false;
        try { comp.step(rng); }
        catch (const std::invalid_argument&) { threw = true; }
        ok(threw, "Defect 1: invalidates on non-child 'beta0' throws on step()");
    }

    // ---- Defect 1b: wrong dependencies key also throws -------------------
    {
        composite_block comp("root_bad_dep");
        comp.add_child(std::make_unique<stub_block>("A", 3.0));
        comp.data().declare_dependencies("not_a_child", {"A"});
        bool threw = false;
        try { comp.step(rng); }
        catch (const std::invalid_argument&) { threw = true; }
        ok(threw, "Defect 1: dependencies on non-child 'not_a_child' throws");
    }

    // ---- Valid dependencies + invalidates: no throw ----------------------
    {
        composite_block comp("root_valid");
        comp.add_child(std::make_unique<stub_block>("A", 1.0));
        comp.add_child(std::make_unique<stub_block>("B", 2.0));
        comp.data().declare_dependencies("B", {"A"});   // B reads A: valid
        comp.data().register_refresher("D", [](const shared_data_t& d) {
            return arma::vec{ d.get("A")[0] };
        });
        comp.data().declare_invalidates("A", {"D"});    // valid child name
        comp.data().refresh_all();
        bool threw = false;
        try { comp.step(rng); comp.step(rng); }
        catch (const std::exception&) { threw = true; }
        ok(!threw, "valid DAG keys: two sweeps run without throwing");
    }

    // ---- Multiple bad keys: error lists ALL of them ----------------------
    {
        composite_block comp("root_multi_bad");
        comp.add_child(std::make_unique<stub_block>("A", 1.0));
        comp.data().register_refresher("D", [](const shared_data_t& d) {
            return arma::vec{ d.get("A")[0] };
        });
        comp.data().declare_invalidates("bad_inval", {"D"});
        comp.data().declare_dependencies("bad_dep", {"A"});
        comp.data().refresh_all();
        std::string msg;
        try { comp.step(rng); }
        catch (const std::invalid_argument& e) { msg = e.what(); }
        bool both = msg.find("bad_inval") != std::string::npos
                 && msg.find("bad_dep")   != std::string::npos;
        ok(both, "error message lists every unmatched key (both bad_inval, bad_dep)");
    }

    // ---- Frozen child does NOT bypass validation -------------------------
    {
        composite_block comp("root_frozen");
        auto child = std::make_unique<stub_block>("A", 1.0);
        child->freeze();                                // whole-block freeze
        comp.add_child(std::move(child));
        comp.data().register_refresher("D", [](const shared_data_t& d) {
            return arma::vec{ d.get("A")[0] };
        });
        comp.data().declare_invalidates("beta0", {"D"});  // wrong key
        comp.data().refresh_all();
        bool threw = false;
        try { comp.step(rng); }
        catch (const std::invalid_argument&) { threw = true; }
        ok(threw, "validation runs even when the only child is frozen");
    }

    // ---- Nested composite: grandchild name is NOT a valid key ------------
    {
        // outer { A, inner { ic } }. A key must name a DIRECT child of the
        // composite it is declared on -- refresh_derived_for is only ever
        // called with direct-child names at each level.
        composite_block outer("outer");
        outer.add_child(std::make_unique<stub_block>("A", 1.0));
        auto inner = std::make_unique<composite_block>("inner");
        inner->add_child(std::make_unique<stub_block>("ic", 2.0));
        outer.add_child(std::move(inner));

        // valid: "inner" IS a direct child of outer
        outer.data().register_refresher("Do", [](const shared_data_t& d) {
            return arma::vec{ d.get("A")[0] };
        });
        outer.data().declare_invalidates("inner", {"Do"});
        outer.data().refresh_all();
        bool ok_ran = true;
        try { outer.step(rng); } catch (const std::exception&) { ok_ran = false; }
        ok(ok_ran, "nested: invalidates on direct-child composite name runs");

        // invalid: "ic" is a GRANDCHILD, not a direct child of outer
        composite_block outer2("outer2");
        outer2.add_child(std::make_unique<stub_block>("A", 1.0));
        auto inner2 = std::make_unique<composite_block>("inner2");
        inner2->add_child(std::make_unique<stub_block>("ic", 2.0));
        outer2.add_child(std::move(inner2));
        outer2.data().register_refresher("X", [](const shared_data_t& d) {
            return arma::vec{ d.get("A")[0] };
        });
        outer2.data().declare_invalidates("ic", {"X"});   // grandchild: invalid
        outer2.data().refresh_all();
        bool threw = false;
        try { outer2.step(rng); }
        catch (const std::invalid_argument&) { threw = true; }
        ok(threw, "nested: invalidates on a grandchild name throws");
    }

    std::printf("\n%s  (%d checks, %d fail)\n",
                fails == 0 ? "[DAG VALIDATION PASS]" : "[DAG VALIDATION FAIL]",
                checks, fails);
    return fails;
}

// Regression test for the default bounds of composite_block's copy band.
//
// Without AI4BAYESCODE_CONTEXT_COPY_MIN_BYTES / AI4BAYESCODE_CONTEXT_COPY_MAX_BYTES
// defined by the includer, composite_block::step() passes a child a fresh copy
// of its context when the context's values take between 36864 and 132000
// bytes (bounds included) in builds for Apple arm64 (__APPLE__ and
// __aarch64__ defined), and views of the stored values for every size
// elsewhere (minimum 1, maximum 0). This test checks the two constants and the
// way a child gets contexts of the sizes on each side of both bounds, on the
// platform it is built for. The constants are detected at compile time, so
// the test also builds against trees without them, and then fails.

#include <armadillo>
#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/shared_data.hpp"
#include "AI4BayesCode/composite_block.hpp"

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

template <class C, class = void>
struct has_copy_band : std::false_type {};
template <class C>
struct has_copy_band<C, std::void_t<decltype(C::context_copy_min_bytes),
                                    decltype(C::context_copy_max_bytes)>>
    : std::true_type {};

// Child that records whether its context X is a view of the stored X.
class way_block : public block_sampler {
public:
    way_block(std::string name, const shared_data_t* d)
        : name_(std::move(name)), cur_(1, arma::fill::zeros), d_(d) {}
    void set_context(const block_context& ctx) override {
        views_.push_back(ctx.at("X").memptr() == d_->get("X").memptr());
    }
    void step(std::mt19937_64& rng) override {
        std::normal_distribution<double> nd(0.0, 1.0);
        cur_[0] = nd(rng);
    }
    const arma::vec& current() const override { return cur_; }
    void set_current(const arma::vec& t) override { cur_ = t; }
    std::size_t dim() const noexcept override { return 1; }
    const std::string& name() const noexcept override { return name_; }
    std::vector<bool> views_;

private:
    std::string name_;
    arma::vec cur_;
    const shared_data_t* d_;
};

template <class C>
static void band_checks() {
    if constexpr (!has_copy_band<C>::value) {
        ok(false, "composite_block::context_copy_min_bytes / context_copy_max_bytes "
                  "are available");
    } else {
#if defined(__APPLE__) && defined(__aarch64__)
        const bool apple_arm64 = true;
        ok(C::context_copy_min_bytes == 36864 && C::context_copy_max_bytes == 132000,
           "Apple arm64: the band is [36864, 132000] bytes");
#else
        const bool apple_arm64 = false;
        ok(C::context_copy_min_bytes == 1 && C::context_copy_max_bytes == 0,
           "not Apple arm64: the band is empty (minimum 1, maximum 0)");
#endif
        struct case_t { std::size_t n; bool copy_on_apple_arm64; };
        const std::vector<case_t> cases = {
            {36864 / 8 - 1, false}, {36864 / 8, true}, {132000 / 8, true},
            {132000 / 8 + 1, false}, {1, false}, {1000000, false}};
        C comp("band");
        shared_data_t& d = comp.data();
        d.set("X", arma::vec(1, arma::fill::ones));
        d.declare_dependencies("w", {"X"});
        auto own = std::make_unique<way_block>("w", &d);
        way_block* w = own.get();
        comp.add_child(std::move(own));
        std::mt19937_64 rng(1);
        for (const auto& c : cases) {
            d.set("X", arma::vec(c.n, arma::fill::value(0.5)));
            const std::size_t first = w->views_.size();
            for (int s = 0; s < 3; ++s) comp.step(rng);
            const bool copy = apple_arm64 && c.copy_on_apple_arm64;
            bool all = w->views_.size() == first + 3;
            for (std::size_t k = first; k < w->views_.size(); ++k)
                if (w->views_[k] == copy) all = false;
            ok(all, "a context of " + std::to_string(8 * c.n) + " bytes gets "
                    + (copy ? "a copy" : "views"));
        }
    }
}

int main() {
    band_checks<composite_block>();
    std::printf("test_fix_composite_copy_band: %d checks, %d failures\n", g_checks, g_fails);
    if (g_fails == 0) {
        std::printf("PASS\n");
        return 0;
    }
    std::printf("FAIL\n");
    return 1;
}

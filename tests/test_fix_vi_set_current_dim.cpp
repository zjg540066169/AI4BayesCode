/*
 * test_fix_vi_set_current_dim.cpp
 *
 * The Gaussian VI blocks keep their variational mean on the unconstrained
 * scale (length K) while current(), set_current() and dim() are on the natural
 * scale. For a dimension-changing constraint such as constraints::simplex
 * (K unconstrained values, K + 1 natural ones) the natural length has to be
 * used for dim() and for the set_current input check, and the vector that
 * set_current stores, unconstrain(theta), has to have length K.
 *
 * For each of mean_field_gaussian_vi_block and full_rank_gaussian_vi_block:
 *   1. dim() and the no-history get_history() row have the natural length;
 *   2. set_current accepts a natural-scale simplex point and current() returns it;
 *   3. set_current rejects a vector of the unconstrained length;
 *   4. set_current rejects a transform whose unconstrain() returns a vector
 *      whose length is not K, and leaves the block unchanged;
 *   5. composite_block current() / set_current() / step() work with the block,
 *      and every published draw is a point of the simplex.
 */

#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/shared_data.hpp"
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/constraints.hpp"
#include "AI4BayesCode/mean_field_gaussian_vi_block.hpp"
#include "AI4BayesCode/full_rank_gaussian_vi_block.hpp"

#include <cmath>
#include <cstdio>
#include <exception>
#include <memory>
#include <random>
#include <stdexcept>

using namespace AI4BayesCode;

namespace {

int checks = 0, failures = 0;

void check(bool ok, const char* what) {
    ++checks;
    if (!ok) { ++failures; std::printf("  FAIL  %s\n", what); }
    else       std::printf("  ok    %s\n", what);
}

// Dirichlet(2, 3, 4, 5) on the 4-category simplex, unconstrained length K = 3.
double lp_dirichlet(const arma::vec& eta, const block_context&, arma::vec* g) {
    const arma::vec a = {2.0, 3.0, 4.0, 5.0};
    return constraints::simplex::wrap(eta, g, [&](const arma::vec& th, arma::vec* gn) {
        if (gn) *gn = (a - 1.0) / th;
        return arma::dot(a - 1.0, arma::log(th));
    });
}

template <class Cfg>
Cfg simplex_cfg() {
    Cfg c;
    c.name = "pi";
    c.initial_unc = arma::zeros(3);
    c.log_density_grad = lp_dirichlet;
    c.constrain = constraints::simplex::constrain;
    c.unconstrain = constraints::simplex::unconstrain;
    return c;
}

// Identity transform on R^3 whose unconstrain drops the last entry.
template <class Cfg>
Cfg short_unconstrain_cfg() {
    Cfg c;
    c.name = "pi";
    c.initial_unc = arma::zeros(3);
    c.log_density_grad = [](const arma::vec& e, const block_context&, arma::vec* g) {
        if (g) *g = -e;
        return -0.5 * arma::dot(e, e);
    };
    c.unconstrain = [](const arma::vec& v) -> arma::vec { return v.head(2); };
    return c;
}

template <class Blk, class Cfg>
void run(const char* label) {
    std::printf("%s\n", label);
    const arma::vec theta = {0.1, 0.2, 0.3, 0.4};

    {   // 1. natural-length dim() and fallback history row
        Blk b(simplex_cfg<Cfg>());
        check(b.dim() == 4 && b.current().n_elem == 4, "dim() equals current().n_elem (4)");
        const auto h = b.get_history();
        check(h.count("pi") == 1 && h.at("pi").n_cols == 4,
              "get_history() without history returns a row of length 4");
    }
    {   // 2. a natural-scale simplex point is accepted
        Blk b(simplex_cfg<Cfg>());
        bool ok = false;
        try {
            b.set_current(theta);
            ok = b.current().n_elem == 4 &&
                 arma::approx_equal(b.current(), theta, "absdiff", 1e-12);
        } catch (const std::exception& e) {
            std::printf("        threw: %s\n", e.what());
        }
        check(ok, "set_current(natural simplex point) is accepted and current() returns it");
    }
    {   // 3. the unconstrained length is rejected
        Blk b(simplex_cfg<Cfg>());
        bool threw = false;
        try { b.set_current(arma::vec{0.2, 0.3, 0.5}); }
        catch (const std::invalid_argument&) { threw = true; }
        catch (const std::exception&) {}
        check(threw, "set_current(length 3) throws std::invalid_argument");
    }
    {   // 4. unconstrain() of the wrong length is rejected; the block is unchanged
        Blk b(short_unconstrain_cfg<Cfg>());
        const arma::vec before = b.current();
        bool threw = false;
        try { b.set_current(arma::vec{1.0, 2.0, 3.0}); }
        catch (const std::invalid_argument&) { threw = true; }
        catch (const std::exception&) {}
        check(threw && b.current().n_elem == 3 &&
              arma::approx_equal(b.current(), before, "absdiff", 0.0),
              "set_current throws when unconstrain(theta) does not have length K");
    }
    {   // 5. composite restore and sweeps
        composite_block comp("m");
        comp.data().set("pi", arma::vec{0.25, 0.25, 0.25, 0.25});
        comp.add_child(std::make_unique<Blk>(simplex_cfg<Cfg>()));
        bool ok = false;
        try {
            const arma::vec cur = comp.current();
            comp.set_current(theta);
            const arma::vec back = comp.current();
            std::mt19937_64 rng(5);
            bool simplex_draws = true;
            for (int t = 0; t < 1200; ++t) {
                comp.step(rng);
                const arma::vec& d = comp.data().get("pi");
                if (d.n_elem != 4 || std::abs(arma::sum(d) - 1.0) > 1e-12 || d.min() <= 0.0)
                    simplex_draws = false;
            }
            ok = cur.n_elem == 4 && back.n_elem == 4 &&
                 arma::approx_equal(back, theta, "absdiff", 1e-12) && simplex_draws;
        } catch (const std::exception& e) {
            std::printf("        threw: %s\n", e.what());
        }
        check(ok, "composite current()/set_current()/step() work and publish simplex draws");
    }
}

} // namespace

int main() {
    std::printf("=== test_fix_vi_set_current_dim ===\n\n");
    run<mean_field_gaussian_vi_block, mean_field_gaussian_vi_block_config>(
        "mean_field_gaussian_vi_block over a 4-category simplex");
    std::printf("\n");
    run<full_rank_gaussian_vi_block, full_rank_gaussian_vi_block_config>(
        "full_rank_gaussian_vi_block over a 4-category simplex");
    std::printf("\n%d checks, %d failures\n", checks, failures);
    if (failures) { std::printf("FAILED\n"); return 1; }
    std::printf("PASSED\n");
    return 0;
}

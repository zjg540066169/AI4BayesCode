/*================================================================================
 *  freeze + predict_at history-alignment regression (2026-07-27).
 *
 *  Bug: a WHOLE-BLOCK-frozen child's step() is skipped by composite_block, so
 *  its per-sweep history append was skipped too. Its history then stalled
 *  while sibling histories kept growing, and any predict_at that iterates the
 *  JOINT history (posterior-predictive over the kept draws -- e.g.
 *  GaussianBartRegression) tripped
 *      "predict_at: inconsistent history sizes (bart=2, sigma=1)".
 *
 *  Fix (block_sampler::record_held_history + composite_block::step): a frozen
 *  child records its HELD (current, unchanged) value each sweep, so every
 *  child's history stays the SAME length -- the frozen parameter is constant
 *  across the kept draws while siblings vary, which is exactly the correct
 *  joint posterior for "hold X fixed, keep sampling the rest".
 *
 *  This test builds two 1-D nuts_blocks in a composite (keep_history = TRUE),
 *  freezes one after some sweeps, and asserts:
 *    (1) both children's histories have the SAME length (= total sweeps), and
 *    (2) the frozen child's post-freeze history entries are all the held value.
 *================================================================================*/

#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/nuts_block.hpp"

#include <armadillo>
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <string>

using AI4BayesCode::block_context;
using AI4BayesCode::composite_block;
using AI4BayesCode::nuts_block;
using AI4BayesCode::nuts_block_config;

namespace {

// 1-D standard normal: lp = -0.5 x^2, grad = -x.
double normal_ld(const arma::vec& x, const block_context&, arma::vec* g) {
    if (g) { g->set_size(1); (*g)[0] = -x[0]; }
    return -0.5 * x[0] * x[0];
}

std::unique_ptr<nuts_block> make_nb(const std::string& nm) {
    nuts_block_config cfg;
    cfg.name = nm;
    cfg.initial_unc = arma::vec{0.0};
    cfg.log_density_grad = &normal_ld;
    cfg.n_warmup_first_call = 5;
    return std::make_unique<nuts_block>(std::move(cfg));
}

}  // namespace

int main() {
    auto root = std::make_unique<composite_block>("root");
    root->add_child(make_nb("a"));
    root->add_child(make_nb("b"));
    root->set_keep_history(true);
    std::mt19937_64 rng(1);

    const int pre = 5, post = 3;
    for (int i = 0; i < pre; ++i) root->step(rng);            // both -> 5
    root->freeze(std::vector<std::string>{"b"});               // whole-block freeze b
    for (int i = 0; i < post; ++i) root->step(rng);           // a -> 8; b HELD -> 8

    const std::size_t expect = static_cast<std::size_t>(pre + post);
    const auto ha = root->child(0).get_history();
    const auto hb = root->child(1).get_history();
    const std::size_t na = ha.at("a").n_rows;
    const std::size_t nb = hb.at("b").n_rows;

    bool ok = true;

    // (1) aligned histories.
    if (na != expect || nb != expect) {
        std::printf("[FAIL] history sizes a=%zu b=%zu (expected %zu each)\n",
                    na, nb, expect);
        ok = false;
    } else {
        std::printf("[ok] histories aligned: a=%zu b=%zu\n", na, nb);
    }

    // (2) frozen b held constant after the freeze point.
    if (nb == expect) {
        const arma::mat& B = hb.at("b");
        const double held = B(pre - 1, 0);   // value at freeze time
        bool constant = true;
        for (std::size_t r = pre; r < nb; ++r)
            if (std::abs(B(r, 0) - held) > 1e-15) constant = false;
        if (!constant) {
            std::printf("[FAIL] frozen b not held constant post-freeze\n");
            ok = false;
        } else {
            std::printf("[ok] frozen b held at %.6f across post-freeze draws\n", held);
        }
    }

    std::printf("%s\n", ok ? "PASS" : "FAILED");
    return ok ? 0 : 1;
}
